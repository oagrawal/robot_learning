"""
Classifier Guidance (CG) Policy.

Wraps a pre-trained FlowPolicy and a pre-trained TrajectoryClassifier.
At inference, the classifier's gradient w.r.t. the proposed action steers
the flow ODE toward higher P(success). No retraining is needed.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque

from imitation.algo.base_algo import BaseAlgo
from imitation.models.trajectory_classifier import TrajectoryClassifier
from imitation.utils.obs_utils import process_obs_dict
from imitation.utils.tensor_utils import recursive_dict_list_tuple_apply


class CGPolicy(nn.Module):
    """
    Classifier-Guided Flow Policy.

    Combines a base FlowPolicy (trained on success demos) with a
    TrajectoryClassifier to add gradient-based guidance at inference.

    The guidance term at each Euler step is:
        alpha * grad_{pred_action} log P(success | obs_window, action_window)
    """

    def __init__(self, base_policy, classifier, alpha=1.0):
        super().__init__()
        self.base_policy = base_policy
        self.classifier = classifier
        self.alpha = alpha

        self.window_size = classifier.window_size
        self.obs_keys = list(classifier.config.observation_config.obs_keys_to_normalize.keys())
        self.obs_keys_to_modality = classifier.config.keys_to_modality

        self.history_obs = deque(maxlen=self.window_size - 1)
        self.history_actions = deque(maxlen=self.window_size - 1)

    def reset(self):
        self.base_policy.reset()
        self.history_obs.clear()
        self.history_actions.clear()

    def to(self, device):
        super().to(device)
        self.base_policy.to(device)
        self.classifier.to(device)
        return self

    def eval(self):
        super().eval()
        self.base_policy.eval()
        self.classifier.eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        self.base_policy.train(mode)
        return self

    def _extract_low_dim_obs(self, raw_obs):
        """Extract the low-dim obs keys the classifier needs from the raw env obs."""
        out = {}
        for k in self.obs_keys:
            val = raw_obs[k]
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val).float()
            if val.dim() == 1:
                val = val.unsqueeze(0)
            out[k] = val
        return out

    @property
    def _history_ready(self):
        """True once we have enough history to fill the classifier window."""
        return len(self.history_obs) >= self.window_size - 1

    def _build_classifier_window(self, current_obs, proposed_action_first):
        """
        Build (obs_window, action_window) for the classifier.

        Only called when history buffer is full (window_size - 1 past steps).

        Args:
            current_obs: dict {key: (B, dim)} low-dim obs tensors
            proposed_action_first: (B, action_dim) first action of predicted chunk
                                   (in whatever space the classifier was trained on)

        Returns:
            obs_window: dict {key: (B, window_size, dim)}
            action_window: (B, window_size, action_dim)
        """
        device = proposed_action_first.device

        obs_list = [{k: v.to(device) for k, v in past_obs.items()}
                    for past_obs in self.history_obs]
        obs_list.append({k: v.to(device) for k, v in current_obs.items()})

        act_list = [past_act.to(device) for past_act in self.history_actions]
        act_list.append(proposed_action_first)

        obs_window = {}
        for k in self.obs_keys:
            obs_window[k] = torch.stack([o[k] for o in obs_list], dim=1)

        action_window = torch.stack(act_list, dim=1)

        return obs_window, action_window

    def _compute_guidance(self, pred_action, current_obs):
        """
        Compute the classifier guidance gradient w.r.t. pred_action.

        Both the flow ODE and the classifier operate in normalized action
        space, so pred_action[:, 0, :] is passed directly — no unnormalization.

        Args:
            pred_action: (B, n_action_steps, action_dim) in normalized action space
            current_obs: dict of raw low-dim obs tensors (classifier normalizes obs internally)

        Returns:
            guidance_grad: (B, n_action_steps, action_dim) gradient in normalized space
        """
        pred_action_grad = pred_action.detach().requires_grad_(True)
        first_action = pred_action_grad[:, 0, :]

        obs_window, action_window = self._build_classifier_window(current_obs, first_action)

        obs_window = process_obs_dict(obs_window, self.obs_keys_to_modality)
        obs_window = recursive_dict_list_tuple_apply(
            obs_window, {torch.Tensor: lambda x: x.float()}
        )

        log_prob = torch.log(self.classifier.predict_prob(obs_window, action_window) + 1e-8)
        grad = torch.autograd.grad(log_prob.sum(), pred_action_grad)[0]

        return grad

    def get_action(self, obs, batched=False):
        self.base_policy.eval()
        self.classifier.eval()

        model = self.base_policy.nets
        processed_obs = self.base_policy.preprocess_obs(obs, batched=batched)
        normalized_obs = model["normalizer"].normalize(processed_obs)
        obs_features = model["obs_encoder"](normalized_obs)

        self.base_policy.obs_queue.append(obs_features)
        while len(self.base_policy.obs_queue) < self.base_policy.n_obs_steps:
            self.base_policy.obs_queue.append(obs_features.clone())

        if len(self.base_policy.action_queue) > 0:
            action = self.base_policy.action_queue.popleft()
            self.base_policy.train()
            return action

        current_low_dim_obs = self._extract_low_dim_obs(obs)
        device = obs_features.device
        use_guidance = self.alpha > 0 and self._history_ready

        global_cond = torch.cat(list(self.base_policy.obs_queue), dim=1)
        pred_action = torch.randn(
            (global_cond.shape[0], self.base_policy.n_action_steps, self.base_policy.action_dim),
            device=device
        )

        delta_t = 1.0 / self.base_policy.num_inference_steps
        timestep = torch.zeros(global_cond.shape[0], device=device)

        for _ in range(self.base_policy.num_inference_steps):
            with torch.no_grad():
                model_output = model["model"](pred_action, timestep, global_cond=global_cond)

            if use_guidance:
                guidance = self._compute_guidance(pred_action, current_low_dim_obs)
                pred_action = pred_action + delta_t * model_output + self.alpha * guidance
            else:
                pred_action = pred_action + delta_t * model_output

            timestep += delta_t

        # Store the first normalized action for history before unnormalizing
        first_action_normalized = pred_action[:, 0, :].detach().clone()

        pred_action = model["normalizer"].unnormalize_by_key(pred_action, 'actions')
        pred_action = pred_action.detach()
        pred_action = pred_action.permute(1, 0, 2)
        self.base_policy.action_queue.extend(pred_action.cpu().numpy())

        executed_action = self.base_policy.action_queue.popleft()

        self.history_obs.append({k: v.detach().cpu() for k, v in current_low_dim_obs.items()})
        self.history_actions.append(first_action_normalized.cpu())

        self.base_policy.train()
        return executed_action
