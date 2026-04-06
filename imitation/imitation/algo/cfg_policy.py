import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from imitation.utils.general_utils import AttrDict
from imitation.algo.base_algo import BaseAlgo
from imitation.utils.lr_scheduler import get_scheduler
from imitation.models.normalizers import DictNormalizer
from imitation.models.obs_nets import ObservationEncoder
from imitation.utils.tensor_utils import recursive_dict_list_tuple_apply
from einops import reduce

from imitation.utils.obs_utils import process_obs_dict
from imitation.utils.torch_utils import replace_bn_with_gn

from collections import deque

from imitation.utils.flow_utils import FlowTimeSampler
from imitation.models.diffusion_mlp_nets import MLPDiffusionHead

class CFGPolicy(BaseAlgo):
    """
    Flow Matching policy with Classifier-Free Guidance (CFG).

    Training:
        - Learns a single conditional velocity field v(x, t | obs, c)
        - c is an 8D vector (one label per action in the chunk):
            c = [1,...,1] for success, c = [0,...,0] for failure
        - 10% of the time, c is replaced with [-1,...,-1] (unconditional dropout)
        - Loss is standard MSE on the flow matching velocity target

    Inference:
        - Queries the model 3x per denoising step: c=success, c=failure, c=unconditional
        - Combines via CFG:
            v_guided = v_uncond + w_succ*(v_succ - v_uncond) - w_fail*(v_fail - v_uncond)
        - w_succ, w_fail, rescale_phi are inference-only hyperparameters
    """

    def __init__(self, config):
        super(CFGPolicy, self).__init__()
        
        self.config = config

        policy_config = config.policy_config
        observation_config = config.observation_config
        keys_to_shapes = config.keys_to_shapes
        
        action_dim = keys_to_shapes['ac_dim']
        
        # --- Networks ---
        self.nets = nn.ModuleDict()

        # Normalizer for obs and actions
        key_to_norm_type = observation_config.obs_keys_to_normalize
        key_to_norm_type['actions'] = config.policy_config.action_normalization_type
        normalizer = DictNormalizer(config.normalization_stats, key_to_norm_type=key_to_norm_type)
        self.nets["normalizer"] = normalizer

        # Observation encoder (ResNet18 for images, passthrough for low-dim)
        obs_encoder = ObservationEncoder(observation_config, keys_to_shapes['obs_shape'], return_dict=False)
        obs_encoder = replace_bn_with_gn(obs_encoder)
        self.nets["obs_encoder"] = obs_encoder

        obs_feature_dim = obs_encoder.output_shape()

        flat_action_dims = action_dim * policy_config.n_action_steps
        global_cond_dim = obs_feature_dim * policy_config.n_obs_steps
        
        # Class conditioning encoder: maps 8D label vector → 64D embedding
        self.cond_embed_dim = getattr(policy_config, 'cond_embed_dim', 64)
        self.nets["c_encoder"] = nn.Sequential(
            nn.Linear(policy_config.n_action_steps, self.cond_embed_dim),
            nn.Mish(),
            nn.Linear(self.cond_embed_dim, self.cond_embed_dim)
        )

        # Flow matching velocity network (conditioned on obs + class embedding)
        model = MLPDiffusionHead(
            input_dim=flat_action_dims + global_cond_dim + policy_config.diffusion_step_embed_dim + self.cond_embed_dim,
            output_dim=flat_action_dims,
            diffusion_step_embed_dim=policy_config.diffusion_step_embed_dim
        )
        self.nets["model"] = model

        # Flow time sampler
        self.flow_time_sampler = FlowTimeSampler(**policy_config.flow_time_sampler_kwargs)

        # --- Dimensions ---
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = policy_config.n_action_steps
        self.n_obs_steps = policy_config.n_obs_steps
        self.num_inference_steps = policy_config.num_inference_steps
        
        # --- CFG parameters ---
        # Training-time: probability of dropping the class label (unconditional training)
        self.uncond_drop_prob = getattr(policy_config, 'uncond_drop_prob', 0.1)
        # Inference-time: guidance weights and rescaling
        self.w_succ = getattr(policy_config, 'w_succ', 2.0)
        self.w_fail = getattr(policy_config, 'w_fail', 1.0)
        self.rescale_phi = getattr(policy_config, 'rescale_phi', 0.0)
        
        self.reset()

    def get_optimizers_and_schedulers(self, **kwargs):
        assert 'num_epochs' in kwargs, 'num_epochs is required'
        assert 'epoch_every_n_steps' in kwargs, 'epoch_every_n_steps is required'

        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-4)
        lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=optimizer,
            num_warmup_steps=1000,
            num_training_steps=kwargs['num_epochs'] * kwargs['epoch_every_n_steps']
        )

        return [optimizer], [lr_scheduler]
        
    def forward(self, batch):
        return None
    
    def psi_t(self, x, x1, t):
        """Conditional flow path: interpolates between noise x and target x1 at time t."""
        t = t[:, None, None]  # (B, 1, 1)
        return (1 - (1 - self.flow_time_sampler.flow_sig_min) * t) * x + t * x1
    
    def compute_loss(self, batch):
        """
        Standard flow matching loss with class-conditional dropout.
        
        The model learns v(x, t | obs, c) where c is the class label vector.
        With probability uncond_drop_prob, c is replaced with -1 (unconditional).
        """
        actions = batch['actions'][:, -1]
        actions = self.nets["normalizer"].normalize_by_key(actions, 'actions')
        B = actions.shape[0]

        # Encode observations
        required_obs = recursive_dict_list_tuple_apply(batch['obs'], {torch.Tensor: lambda x: x[:, :self.n_obs_steps, ...].clone()})
        normalized_obs = self.nets["normalizer"].normalize(required_obs)
        obs_features = self.nets["obs_encoder"](normalized_obs)
        global_cond = obs_features.reshape(B, -1)

        # Process class labels
        c = batch['c'].float()
        if c.dim() == 1:
            c = c.unsqueeze(0)
        assert c.shape[-1] == self.n_action_steps, (
            f"batch['c'] must have last dim {self.n_action_steps}, got {c.shape[-1]}"
        )

        # Unconditional dropout: replace label with -1 vector
        if self.training and self.uncond_drop_prob > 0:
            mask = (torch.rand(B, device=c.device) < self.uncond_drop_prob).float()
            c = c * (1 - mask[:, None]) + (-1.0) * mask[:, None]

        # Encode class label and concatenate with obs conditioning
        c_emb = self.nets["c_encoder"](c)
        global_cond = torch.cat([global_cond, c_emb], dim=-1)

        # Flow matching: sample time, interpolate, predict velocity
        noise = torch.randn_like(actions, device=actions.device)
        timesteps = self.flow_time_sampler.sample_fm_time(B).to(actions.device)
        psi_t = self.psi_t(noise, actions, timesteps)

        v_psi = self.nets["model"](psi_t, timesteps, global_cond=global_cond)
        d_psi = actions - (1 - self.flow_time_sampler.flow_sig_min) * noise

        # MSE loss
        losses = AttrDict()
        loss = F.mse_loss(v_psi, d_psi, reduction='none')
        loss = reduce(loss, 'b t a -> b', 'mean')
        losses.mse = loss.mean()
        losses.total = losses.mse
        return losses
    
    def post_step_update(self):
        pass

    def preprocess_obs(self, obs, batched=False):
        """Convert obs dict from numpy/mixed types to batched float tensors on device."""
        obs = process_obs_dict(obs, self.config.keys_to_modality)
        transform = {
            torch.Tensor: lambda x: x.float().to(self.device),
            np.ndarray: lambda x: torch.from_numpy(x).float().to(self.device),
            type(None): lambda x: x,
        }
        if not batched:
            transform[torch.Tensor] = lambda x: x[None].float().to(self.device)
            transform[np.ndarray] = lambda x: torch.from_numpy(x)[None].float().to(self.device)
        return recursive_dict_list_tuple_apply(obs, transform)

    def _build_cfg_conditioning(self, global_cond, device):
        """
        Build the three CFG conditioning vectors (unconditional, success, failure)
        and stack them for a single batched forward pass.
        
        Returns:
            cond_all: (3*B, cond_dim) stacked conditioning
        """
        B = global_cond.shape[0]
        
        c_uncond = torch.full((B, self.n_action_steps), -1.0, device=device)
        c_succ = torch.ones((B, self.n_action_steps), device=device)
        c_fail = torch.zeros((B, self.n_action_steps), device=device)

        c_uncond_emb = self.nets["c_encoder"](c_uncond)
        c_succ_emb = self.nets["c_encoder"](c_succ)
        c_fail_emb = self.nets["c_encoder"](c_fail)
        
        cond_uncond = torch.cat([global_cond, c_uncond_emb], dim=-1)
        cond_succ = torch.cat([global_cond, c_succ_emb], dim=-1)
        cond_fail = torch.cat([global_cond, c_fail_emb], dim=-1)
        
        return torch.cat([cond_uncond, cond_succ, cond_fail], dim=0)

    def _cfg_denoise_loop(self, pred_action, cond_all, B, device):
        """
        Run the flow matching ODE with CFG guidance.
        
        At each step:
            1. Query the model with all 3 conditions in one batched pass
            2. Apply CFG formula: v = v_uncond + w_succ*(v_succ - v_uncond) - w_fail*(v_fail - v_uncond)
            3. Optionally rescale to match v_succ norm (rescale_phi)
            4. Euler step: x += dt * v
        """
        model = self.nets
        delta_t = 1.0 / self.num_inference_steps
        timestep = torch.zeros(B, device=device)
        
        for _ in range(self.num_inference_steps):
            pred_all = torch.cat([pred_action] * 3, dim=0)
            t_all = torch.cat([timestep] * 3, dim=0)
            
            model_output = model["model"](pred_all, t_all, global_cond=cond_all)
            v_uncond, v_succ, v_fail = torch.chunk(model_output, 3, dim=0)
            
            # CFG guidance formula
            v_guided = v_uncond + self.w_succ * (v_succ - v_uncond) - self.w_fail * (v_fail - v_uncond)

            # Optional: rescale guided velocity to match success velocity norm
            if self.rescale_phi > 0.0:
                norm_guided = torch.linalg.norm(v_guided, dim=-1, keepdim=True)
                norm_succ = torch.linalg.norm(v_succ, dim=-1, keepdim=True)
                v_guided_rescaled = v_guided * (norm_succ / (norm_guided + 1e-6))
                v_guided = self.rescale_phi * v_guided_rescaled + (1.0 - self.rescale_phi) * v_guided

            # Euler integration step
            pred_action += delta_t * v_guided
            timestep += delta_t
        
        return pred_action

    @torch.no_grad()
    def get_action(self, obs, batched=False):
        """Get action for environment interaction (with obs history queue)."""
        self.eval()

        obs = self.preprocess_obs(obs, batched=batched)
        obs = self.nets["normalizer"].normalize(obs)
        obs_features = self.nets["obs_encoder"](obs)
        
        # Maintain observation history window
        self.obs_queue.append(obs_features)
        while len(self.obs_queue) < self.n_obs_steps:
            self.obs_queue.append(obs_features.clone())
        
        # Return cached action if available
        if len(self.action_queue) > 0:
            action = self.action_queue.popleft()
            self.train()
            return action
        
        # Run CFG denoising to generate new action chunk
        global_cond = torch.cat(list(self.obs_queue), dim=1)
        B = global_cond.shape[0]
        pred_action = torch.randn((B, self.n_action_steps, self.action_dim), device=obs_features.device)
        
        cond_all = self._build_cfg_conditioning(global_cond, obs_features.device)
        pred_action = self._cfg_denoise_loop(pred_action, cond_all, B, obs_features.device)
        
        # Unnormalize and queue actions
        pred_action = self.nets["normalizer"].unnormalize_by_key(pred_action, 'actions')
        pred_action = pred_action.permute(1, 0, 2)
        self.action_queue.extend(pred_action.cpu().numpy())

        self.train()
        action = self.action_queue.popleft()
        return action

    def to(self, device):
        super().to(device)

    def reset(self):
        """Reset obs and action queues (call at start of each rollout)."""
        self.obs_queue = deque(maxlen=self.n_obs_steps)
        self.action_queue = deque(maxlen=self.n_action_steps)

    @torch.no_grad()
    def get_output(self, obs):
        """Get batched action output (for action distribution visualization, no obs queue)."""
        self.eval()

        obs = self.preprocess_obs(obs, batched=True)
        obs = self.nets["normalizer"].normalize(obs)
        obs_features = self.nets["obs_encoder"](obs)

        B = obs_features.shape[0]
        global_cond = obs_features.reshape(B, -1)
        pred_action = torch.randn((B, self.n_action_steps, self.action_dim), device=obs_features.device)
    
        cond_all = self._build_cfg_conditioning(global_cond, obs_features.device)
        pred_action = self._cfg_denoise_loop(pred_action, cond_all, B, obs_features.device)
        
        pred_action = self.nets["normalizer"].unnormalize_by_key(pred_action, 'actions')

        self.train()
        return pred_action.cpu().numpy()
