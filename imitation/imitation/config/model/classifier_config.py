import os

from collections import OrderedDict

import torch.nn as nn

from imitation.data.classifier_dataset import ClassifierDataset
from imitation.models.image_nets import ResNet18, SpatialSoftmax
from imitation.models.obs_nets import VisionCore
from imitation.utils.general_utils import AttrDict

# HDF5s and transition JSON live next to this package: imitation/imitation/data/
_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
# Requires a success (positive) dataset at square_d0.hdf5 with the same obs/action layout
# as the failure file (see observation_config keys).

# Match FlowPolicy: 2 obs frames, 8 action chunk (see flow_policy_config.py)
classifier_config = AttrDict(
    n_obs_steps=2,
    action_chunk_size=8,
)

train_config = AttrDict(
    output_dir="~/robot_learning/experiments",
    batch_size=64,
    num_epochs=100,
    lr=1e-4,
    weight_decay=1e-4,
    val_every_n_epochs=5,
    save_every_n_epochs=10,
    num_workers=4,
    focal_gamma=2.0,
    eval_test_every_n_epochs=0,
)

data_config = AttrDict(
    data=[
        os.path.join(_DATA_DIR, "square_d0.hdf5"),
        os.path.join(_DATA_DIR, "failure_185_negative.hdf5"),
    ],
    dataset_class=ClassifierDataset,
    dataset_kwargs=dict(
        num_pos=1,
        n_obs_steps=2,
        action_chunk_size=8,
        max_demo_len=None,
        # three_way_hard_val split (see ClassifierDataset docstring):
        #   pos: 80% train / 10% val-pool / 10% test (disjoint success demos).
        #   neg: JSON demos -> val (windows in [start_t, end_t] only); non-JSON
        #        failures shuffled -> 80% train / 20% test.
        split_strategy='three_way_hard_val',
        n_val_demos_per_class=None,
        n_test_demos_per_class=None,
        ablation_mode="action_and_state",
        hard_eval_seed=42,
        hard_val_json=os.path.join(_DATA_DIR, "transition_val_example.json"),
        hard_test_json=None,
        # JSON val failures are never in the train set under 'three_way_hard_val'
        # (they are placed directly into val), so this flag is redundant there.
        exclude_hard_json_failures_from_train=True,
    ),
)

observation_config = AttrDict(
    obs=OrderedDict(
        low_dim=[
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        rgb=[
            "agentview_image",
            "robot0_eye_in_hand_image",
        ],
        depth=[],
    ),
    obs_keys_to_normalize={
        "robot0_eef_pos": 'gaussian',
        "robot0_eef_quat": 'gaussian',
        "robot0_gripper_qpos": 'gaussian',
    },
    encoder=AttrDict(
        low_dim=AttrDict(
            core_class=None,
            core_kwargs=dict(
                output_dim=16,
                hidden_units=[16],
                activation=nn.LeakyReLU(0.2),
                output_activation=nn.LeakyReLU(0.2),
            ),
        ),
        rgb=AttrDict(
            core_class=VisionCore,
            core_kwargs=dict(
                backbone_class=ResNet18,
                backbone_kwargs=None,
                feature_dim=64,
                pool_class=SpatialSoftmax,
                pool_kwargs=dict(
                    num_kp=32,
                    learnable_temperature=False,
                    temperature=1.0,
                    noise_std=0.0,
                ),
            ),
        ),
        depth=AttrDict(
            core_class=None,
            core_kwargs=dict(),
        ),
    )
)

config = AttrDict(
    classifier_config=classifier_config,
    train_config=train_config,
    data_config=data_config,
    observation_config=observation_config,
)
