from imitation.utils.general_utils import AttrDict
from imitation.models.image_nets import ResNet18, SpatialSoftmax
from imitation.models.obs_nets import VisionCore
from imitation.data.classifier_dataset import ClassifierDataset
import torch.nn as nn
from collections import OrderedDict

classifier_config = AttrDict(
    window_size=3,
)

train_config = AttrDict(
    output_dir="~/robot_learning/experiments",
    batch_size=256,
    num_epochs=100,
    lr=1e-4,
    weight_decay=1e-4,
    val_every_n_epochs=5,
    save_every_n_epochs=10,
    num_workers=10,
)

data_config = AttrDict(
    data=[
        "./data/square_d0.hdf5",
        "./data/failure_dense_labeled_1000.hdf5",
    ],
    dataset_class=ClassifierDataset,
    dataset_kwargs=dict(
        num_pos=1,
        window_size=3,
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
