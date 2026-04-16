"""
Classifier Guidance (CG) Policy.

Wraps a pretrained FlowPolicy and TrajectoryClassifier. At inference, adds
grad w.r.t. the full predicted action chunk toward higher P(success | obs, actions).
"""

import torch
import torch.nn as nn
from collections import deque

from imitation.models.trajectory_classifier import TrajectoryClassifier
from imitation.utils.obs_utils import process_obs_dict
from imitation.utils.tensor_utils import recursive_dict_list_tuple_apply


def _classifier_obs_keys(observation_config):
    keys = []
    for modality in ("low_dim", "rgb", "depth"):
        keys.extend(list(observation_config.obs.get(modality) or []))
    return keys


class CGPolicy(nn.Module):
    """
    Classifier-guided flow policy. Guidance:
        alpha * grad_{pred_action} log P(success | obs_window, action_chunk)
    where obs_window matches classifier n_obs_steps and action_chunk matches n_action_steps.
    """

    def __init__(self, base_policy, classifier, alpha=1.0):
        super().__init__()
        self.base_policy = base_policy
        self.classifier = classifier
        self.alpha = alpha

        self.n_obs_steps = classifier.n_obs_steps
        self.action_chunk_size = classifier.action_chunk_size
        if self.action_chunk_size != base_policy.n_action_steps:
            raise ValueError(
                f"classifier action_chunk_size {self.action_chunk_size} != "
                f"base n_action_steps {base_policy.n_action_steps}"
            )

        self.classifier_obs_keys = _classifier_obs_keys(classifier.config.observation_config)
        self.obs_keys_to_modality = classifier.config.keys_to_modality

        self.history_obs = deque(maxlen=max(1, self.n_obs_steps - 1))

    def reset(self):
        self.base_policy.reset()
        self.history_obs.clear()

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

    def _preprocess_classifier_obs(self, obs, batched=False):
        """Same preprocessing as FlowPolicy (scale, layout), subset to classifier keys."""
        proc = self.base_policy.preprocess_obs(obs, batched=batched)
        return {k: proc[k] for k in self.classifier_obs_keys}

    @property
    def _history_ready(self):
        return len(self.history_obs) >= self.n_obs_steps - 1

    def _build_classifier_inputs(self, current_obs_proc, pred_action_norm):
        """
        current_obs_proc: dict key -> (B, ...) single frame (already preprocessed)
        pred_action_norm: (B, n_action_steps, action_dim) normalized chunk
        """
        device = pred_action_norm.device
        obs_list = [{k: v.to(device) for k, v in past.items()} for past in self.history_obs]
        obs_list.append({k: v.to(device) for k, v in current_obs_proc.items()})

        obs_window = {}
        for k in self.classifier_obs_keys:
            obs_window[k] = torch.stack([o[k] for o in obs_list], dim=1)

        return obs_window, pred_action_norm

    def _compute_guidance(self, pred_action, current_obs_proc):
        pred_action_grad = pred_action.detach().requires_grad_(True)
        obs_window, act_chunk = self._build_classifier_inputs(current_obs_proc, pred_action_grad)

        obs_window = process_obs_dict(obs_window, self.obs_keys_to_modality)
        obs_window = recursive_dict_list_tuple_apply(
            obs_window, {torch.Tensor: lambda x: x.float()}
        )

        log_prob = torch.log(self.classifier.predict_prob(obs_window, act_chunk) + 1e-8)
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

        current_classifier_obs = self._preprocess_classifier_obs(obs, batched=batched)
        device = obs_features.device
        use_guidance = self.alpha > 0 and self._history_ready

        global_cond = torch.cat(list(self.base_policy.obs_queue), dim=1)
        pred_action = torch.randn(
            (global_cond.shape[0], self.base_policy.n_action_steps, self.base_policy.action_dim),
            device=device,
        )

        delta_t = 1.0 / self.base_policy.num_inference_steps
        timestep = torch.zeros(global_cond.shape[0], device=device)

        for _ in range(self.base_policy.num_inference_steps):
            with torch.no_grad():
                model_output = model["model"](pred_action, timestep, global_cond=global_cond)

            if use_guidance:
                guidance = self._compute_guidance(pred_action, current_classifier_obs)
                pred_action = pred_action + delta_t * model_output + self.alpha * guidance
            else:
                pred_action = pred_action + delta_t * model_output

            timestep += delta_t

        pred_action = model["normalizer"].unnormalize_by_key(pred_action, 'actions')
        pred_action = pred_action.detach()
        pred_action = pred_action.permute(1, 0, 2)
        self.base_policy.action_queue.extend(pred_action.cpu().numpy())

        executed_action = self.base_policy.action_queue.popleft()

        self.history_obs.append(
            {k: v.detach().cpu() for k, v in current_classifier_obs.items()}
        )

        self.base_policy.train()
        return executed_action
