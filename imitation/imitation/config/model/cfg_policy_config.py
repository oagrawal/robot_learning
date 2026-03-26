import os

from imitation.utils.general_utils import AttrDict
from imitation.algo.cfg_policy import CFGPolicy
from imitation.models.image_nets import ResNet18, SpatialSoftmax
from imitation.models.obs_nets import VisionCore, LowDimCore
from imitation.data.cfg_dataset import CFGSequenceDataset
import torch.nn as nn
import diffusers
from collections import OrderedDict
import robosuite as suite
import mimicgen

window_size = 2
action_horizon = 8

train_config = AttrDict(
    output_dir="~/robot_learning/experiments",
    batch_size=256,
    num_epochs=1000,
    epoch_every_n_steps=500,
    log_every_n_epochs=1,
    val_every_n_epochs=25,
    save_every_n_epochs=50,
    eval_every_n_epochs=50,
    seed=1
)

data_config = AttrDict(
    data=[
        "./data/success.hdf5",
        "./data/failure.hdf5"
    ],
    dataset_class=CFGSequenceDataset,
    dataset_kwargs=dict(
        dataset_keys=['actions'],
        window_size=window_size,
        action_horizon=action_horizon,
        num_pos=1, # Important! Tells CFGSequenceDataset that the first path is positive data
    ),
    num_workers=20
)

policy_config = AttrDict(
    policy_class=CFGPolicy,
    n_obs_steps=window_size,
    n_action_steps=action_horizon,
    diffusion_step_embed_dim=128,
    down_dims=[256,512,1024],
    kernel_size=5,
    n_groups=8,
    cond_predict_scale=True,
    pos_embedding_period=100,

    flow_time_sampler_kwargs=dict(
        flow_sampling='uniform',
        flow_alpha=1.5,
        flow_beta=1.,
        flow_sig_min=0.001,
    ),
    num_inference_steps=50,
    action_normalization_type='gaussian',

    # CFG Parameters
    w_succ=2.0,
    w_fail=0.5,
    uncond_drop_prob=0.1
)

observation_config = AttrDict(
    obs = OrderedDict(
        low_dim = [
            "robot0_eef_pos",       
            "robot0_eef_quat",      
            "robot0_gripper_qpos",  
        ],
        rgb = [
            "agentview_image",      
            "robot0_eye_in_hand_image", 
        ],
        depth = [],
    ),
    obs_keys_to_normalize = {
        "robot0_eef_pos": 'gaussian',
        "robot0_eef_quat": 'gaussian',
        "robot0_gripper_qpos": 'gaussian',
    },
    encoder = AttrDict(
        low_dim = AttrDict(
            core_class=None,
            core_kwargs=dict(
                output_dim=16,
                hidden_units=[16],
                activation=nn.LeakyReLU(0.2),
                output_activation=nn.LeakyReLU(0.2),
            ),
        ),
        rgb = AttrDict(
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
        depth = AttrDict(
            core_class=None,
            core_kwargs=dict(),
        ),
    )
)

from imitation.evaluators.robosuite_evaluator import RobosuiteEvaluator

# Square_D0
env_config = AttrDict(
    env_name="Square_D0",
    robots="Panda",
    controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
    has_renderer=False,
    has_offscreen_renderer=True,
    reward_shaping=False,
    use_camera_obs=True,
    camera_names=["agentview", "robot0_eye_in_hand"],
    camera_heights=84,
    camera_widths=84,
)


# Stack_D0
# env_config = AttrDict(
#     env_name="Stack_D0",
#     robots="Panda",
#     controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
#     has_renderer=False,
#     has_offscreen_renderer=True,
#     reward_shaping=False,
#     use_camera_obs=True,
#     camera_names=["agentview", "robot0_eye_in_hand"],
#     camera_heights=84,
#     camera_widths=84,
# )

evaluator_config = AttrDict(
    evaluator=RobosuiteEvaluator,
    env_config=env_config,
    n_rollouts=30,
    max_steps=400,
    save_video=True,
    video_folder="../../rollout_videos_square_d0"
)

config = AttrDict(
    train_config=train_config,
    data_config=data_config,
    observation_config=observation_config,
    policy_config=policy_config,
    evaluator_config=evaluator_config,
)
