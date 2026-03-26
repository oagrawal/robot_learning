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

    def __init__(self, config):
        super(CFGPolicy, self).__init__()
        
        self.config = config

        policy_config = config.policy_config
        observation_config = config.observation_config
        keys_to_shapes = config.keys_to_shapes
        
        action_dim = keys_to_shapes['ac_dim']
        
        self.nets = nn.ModuleDict()
        key_to_norm_type = observation_config.obs_keys_to_normalize
        key_to_norm_type['actions'] = config.policy_config.action_normalization_type
        normalizer = DictNormalizer(config.normalization_stats, key_to_norm_type=key_to_norm_type)
        self.nets["normalizer"] = normalizer

        obs_encoder = ObservationEncoder(observation_config, keys_to_shapes['obs_shape'], return_dict=False)
        obs_encoder = replace_bn_with_gn(obs_encoder)
        self.nets["obs_encoder"] = obs_encoder

        obs_feature_dim = obs_encoder.output_shape()

        flat_action_dims = action_dim*policy_config.n_action_steps
        global_cond_dim = obs_feature_dim * policy_config.n_obs_steps
        
        # +1 input dim to support the class token 'c' concatenated perfectly to the end computation
        model = MLPDiffusionHead(
            input_dim=flat_action_dims + global_cond_dim + policy_config.diffusion_step_embed_dim + 1,
            output_dim=flat_action_dims,
            diffusion_step_embed_dim=policy_config.diffusion_step_embed_dim
        )

        self.nets["model"] = model

        self.flow_time_sampler = FlowTimeSampler(**policy_config.flow_time_sampler_kwargs)

        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = policy_config.n_action_steps
        self.n_obs_steps = policy_config.n_obs_steps
        self.num_inference_steps = policy_config.num_inference_steps
        
        # Pull CFG parameters gracefully from config
        self.uncond_drop_prob = getattr(policy_config, 'uncond_drop_prob', 0.1)
        self.w_succ = getattr(policy_config, 'w_succ', 2.0)
        self.w_fail = getattr(policy_config, 'w_fail', 1.0)
        
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
    
    def psi_t(self, x: torch.FloatTensor, x1: torch.FloatTensor, t: torch.FloatTensor) -> torch.FloatTensor:
        """Conditional Flow"""
        t = t[:, None, None]  # (B, 1, 1)
        return (1 - (1 - self.flow_time_sampler.flow_sig_min) * t) * x + t * x1
    
    def compute_loss(self, batch):
        actions = batch['actions'][:, -1]
        actions = self.nets["normalizer"].normalize_by_key(actions, 'actions')
        B = actions.shape[0]

        # obs as global cond
        required_obs = recursive_dict_list_tuple_apply(batch['obs'], {torch.Tensor: lambda x: x[:, :self.n_obs_steps, ...].clone()})
        normalized_obs = self.nets["normalizer"].normalize(required_obs)

        obs_features = self.nets["obs_encoder"](normalized_obs)
        global_cond = obs_features.reshape(B, -1)

        c = batch['c'] # Shape (B, 1)
        
        # CFG Unconditional Drop 
        # With 10% prob during training, overwrite C with -1.0 so network learns unconditional manifold
        if self.training and self.uncond_drop_prob > 0:
            mask = (torch.rand(B, 1, device=c.device) < self.uncond_drop_prob).float()
            c = c * (1 - mask) + (-1.0) * mask
            
        # Push 'C' seamlessly onto the global_cond
        global_cond = torch.cat([global_cond, c], dim=-1)

        noise = torch.randn_like(actions, device=actions.device)
        timesteps = self.flow_time_sampler.sample_fm_time(B).to(actions.device)
        psi_t = self.psi_t(noise, actions, timesteps)

        v_psi = self.nets["model"](psi_t, timesteps, global_cond=global_cond)
        d_psi = actions - (1 - self.flow_time_sampler.flow_sig_min) * noise

        losses = AttrDict()
        loss = F.mse_loss(v_psi, d_psi, reduction='none')
        
        loss = reduce(loss, 'b t a -> b', 'mean')
        
        losses.mse = loss.mean()
        losses.total = losses.mse
        return losses
    
    def post_step_update(self):
        pass

    def preprocess_obs(self, obs, batched=False):
        obs = process_obs_dict(obs, self.config.keys_to_modality)

        if not batched:
            obs = recursive_dict_list_tuple_apply(
                obs,
                {
                    torch.Tensor: lambda x: x[None].float().to(self.device),
                    np.ndarray: lambda x: torch.from_numpy(x)[None].float().to(self.device),
                    type(None): lambda x: x,
                }
            )
        else:
            obs = recursive_dict_list_tuple_apply(
                obs,
                {
                    torch.Tensor: lambda x: x.float().to(self.device),
                    np.ndarray: lambda x: torch.from_numpy(x).float().to(self.device),
                    type(None): lambda x: x,
                }
            )

        return obs

    @torch.no_grad()
    def get_action(self, obs, batched=False):
        self.eval()

        model = self.nets
        obs = self.preprocess_obs(obs, batched=batched)
        obs = self.nets["normalizer"].normalize(obs)
        obs_features = model["obs_encoder"](obs)
        
        self.obs_queue.append(obs_features)
        while len(self.obs_queue) < self.n_obs_steps:
            self.obs_queue.append(obs_features.clone())
        
        if len(self.action_queue) > 0:
            action = self.action_queue.popleft()
            self.train()
            return action
        
        global_cond = torch.cat(list(self.obs_queue), dim=1)
        B = global_cond.shape[0]
        pred_action = torch.randn((B, self.n_action_steps, self.action_dim), device=obs_features.device)
        
        # Prepare 3 conditional vectors: unconditional (-1), success (1), fail (0)
        c_uncond = torch.ones((B, 1), device=obs_features.device) * -1.0
        c_succ = torch.ones((B, 1), device=obs_features.device) * 1.0
        c_fail = torch.ones((B, 1), device=obs_features.device) * 0.0
        
        cond_uncond = torch.cat([global_cond, c_uncond], dim=-1)
        cond_succ = torch.cat([global_cond, c_succ], dim=-1)
        cond_fail = torch.cat([global_cond, c_fail], dim=-1)
        
        # Stack so we can query the network in one batched forward pass for speed
        cond_all = torch.cat([cond_uncond, cond_succ, cond_fail], dim=0)
        
        delta_t = 1.0 / self.num_inference_steps
        timestep = torch.zeros(B, device=obs_features.device)
        
        for _ in range(self.num_inference_steps):
            pred_all = torch.cat([pred_action] * 3, dim=0)
            t_all = torch.cat([timestep] * 3, dim=0)
            
            model_output = model["model"](pred_all, t_all, global_cond=cond_all)
            
            # Split the stacked predictions back into the three scenarios
            v_uncond, v_succ, v_fail = torch.chunk(model_output, 3, dim=0)
            
            # Formulate the explicit gradient repel
            v_guided = v_uncond + self.w_succ * (v_succ - v_uncond) - self.w_fail * (v_fail - v_uncond)

            pred_action += delta_t * v_guided
            timestep += delta_t
        
        # unnormalize actions
        pred_action = self.nets["normalizer"].unnormalize_by_key(pred_action, 'actions')
        pred_action = pred_action.permute(1, 0, 2)
        self.action_queue.extend(pred_action.cpu().numpy())

        self.train()
        action = self.action_queue.popleft()
        return action

    def to(self, device):
        super().to(device)

    def reset(self):
        self.obs_queue = deque(maxlen=self.n_obs_steps)
        self.action_queue = deque(maxlen=self.n_action_steps)

    @torch.no_grad()
    def get_output(self, obs):
        self.eval()

        model = self.nets
    
        obs = self.preprocess_obs(obs, batched=True)
        obs = self.nets["normalizer"].normalize(obs)
        obs_features = model["obs_encoder"](obs)

        B = obs_features.shape[0]
        global_cond = obs_features.reshape(B, -1)
        pred_action = torch.randn((B, self.n_action_steps, self.action_dim), device=obs_features.device)
    
        c_uncond = torch.ones((B, 1), device=obs_features.device) * -1.0
        c_succ = torch.ones((B, 1), device=obs_features.device) * 1.0
        c_fail = torch.ones((B, 1), device=obs_features.device) * 0.0
        
        cond_uncond = torch.cat([global_cond, c_uncond], dim=-1)
        cond_succ = torch.cat([global_cond, c_succ], dim=-1)
        cond_fail = torch.cat([global_cond, c_fail], dim=-1)
        
        cond_all = torch.cat([cond_uncond, cond_succ, cond_fail], dim=0)

        delta_t = 1.0 / self.num_inference_steps
        timestep = torch.zeros(B, device=obs_features.device)
        
        for _ in range(self.num_inference_steps):
            pred_all = torch.cat([pred_action] * 3, dim=0)
            t_all = torch.cat([timestep] * 3, dim=0)
            
            model_output = model["model"](pred_all, t_all, global_cond=cond_all)
            
            v_uncond, v_succ, v_fail = torch.chunk(model_output, 3, dim=0)
            
            v_guided = v_uncond + self.w_succ * (v_succ - v_uncond) - self.w_fail * (v_fail - v_uncond)

            pred_action += delta_t * v_guided
            timestep += delta_t
        
        pred_action = self.nets["normalizer"].unnormalize_by_key(pred_action, 'actions')

        self.train()
        return pred_action.cpu().numpy()
