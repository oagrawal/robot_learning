from imitation.utils.general_utils import AttrDict
from imitation.models.image_nets import ResNet18, SpatialSoftmax
from imitation.models.obs_nets import VisionCore
from imitation.data.classifier_dataset import ClassifierDataset
import torch.nn as nn
from collections import OrderedDict

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
        "./data/square_d0.hdf5",
        "./data/failure_dense_labeled_1000.hdf5",
    ],
    dataset_class=ClassifierDataset,
    dataset_kwargs=dict(
        num_pos=1,
        n_obs_steps=2,
        action_chunk_size=8,
        max_demo_len=None,
        n_val_demos_per_class=30,
        n_test_demos_per_class=30,
        ablation_mode="action_and_state",
        hard_eval_seed=42,
        hard_val_json=None,
        hard_test_json=None,
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
