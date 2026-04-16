import torch
import torch.nn as nn

from imitation.models.obs_nets import ObservationEncoder
from imitation.models.normalizers import DictNormalizer
from imitation.models.base_nets import MLP
from imitation.utils.torch_utils import replace_bn_with_gn


class TrajectoryClassifier(nn.Module):
    """
    Binary classifier: P(success) from n_obs_steps observation frames and an
    action_chunk_size-step action chunk (same layout as FlowPolicy at inference).

    obs_encoder: ResNet18 + SpatialSoftmax per RGB key + passthrough low_dim
    head: MLP on concat(obs_features_flat, actions_flat) -> logit
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        classifier_config = config.classifier_config
        observation_config = config.observation_config
        keys_to_shapes = config.keys_to_shapes

        self.n_obs_steps = int(classifier_config.n_obs_steps)
        self.action_chunk_size = int(classifier_config.action_chunk_size)
        action_dim = keys_to_shapes['ac_dim']

        self.nets = nn.ModuleDict()

        key_to_norm_type = dict(observation_config.obs_keys_to_normalize)
        normalizer = DictNormalizer(config.normalization_stats, key_to_norm_type=key_to_norm_type)
        self.nets["normalizer"] = normalizer

        obs_encoder = ObservationEncoder(
            observation_config, keys_to_shapes['obs_shape'], return_dict=False
        )
        obs_encoder = replace_bn_with_gn(obs_encoder)
        self.nets["obs_encoder"] = obs_encoder

        obs_feature_dim = obs_encoder.output_shape()
        input_dim = self.n_obs_steps * obs_feature_dim + self.action_chunk_size * action_dim
        self.nets["head"] = MLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_units=[256, 128],
            activation=nn.ReLU(),
            output_activation=None,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, obs_dict, actions):
        """
        Args:
            obs_dict: {key: (B, n_obs_steps, ...)} per key
            actions:  (B, action_chunk_size, action_dim) — normalized actions
        Returns:
            logits: (B, 1)
        """
        B = actions.shape[0]
        normalized_obs = self.nets["normalizer"].normalize(obs_dict)
        obs_features = self.nets["obs_encoder"](normalized_obs)
        obs_flat = obs_features.reshape(B, -1)
        act_flat = actions.reshape(B, -1)
        x = torch.cat([obs_flat, act_flat], dim=-1)
        return self.nets["head"](x)

    def predict_prob(self, obs_dict, actions):
        return torch.sigmoid(self.forward(obs_dict, actions))

    def save(self, path):
        torch.save({'config': self.config, 'state_dict': self.state_dict()}, path)

    @staticmethod
    def load(path, device='cpu'):
        data = torch.load(path, weights_only=False, map_location=device)
        model = TrajectoryClassifier(data['config'])
        model.load_state_dict(data['state_dict'])
        return model
