"""
Classifier Guidance (CG) policy config.

Inference-only: specifies paths to a pre-trained FlowPolicy and a
pre-trained TrajectoryClassifier, plus the guidance scale alpha.
"""

import os
import robosuite as suite
from imitation.utils.general_utils import AttrDict
from imitation.evaluators.robosuite_evaluator import RobosuiteEvaluator

cg_config = AttrDict(
    base_policy_ckpt="~/robot_learning/experiments/unet_base/weights/weights_ep1000.pth",
    classifier_ckpt="~/robot_learning/experiments/classifier_v1/best_classifier.pth",
    alpha=1.0,
)

env_config = AttrDict(
    env_name="Square_D0",
    robots="Panda",
    controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
    has_renderer=False,
    has_offscreen_renderer=True,
    reward_shaping=True,
    use_camera_obs=True,
    camera_names=["agentview", "robot0_eye_in_hand"],
    camera_heights=84,
    camera_widths=84,
)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

evaluator_config = AttrDict(
    evaluator=RobosuiteEvaluator,
    env_config=env_config,
    n_rollouts=30,
    max_steps=400,
    save_video=True,
    video_folder=os.path.join(_PROJECT_ROOT, "rollout_videos"),
)

config = AttrDict(
    cg_config=cg_config,
    evaluator_config=evaluator_config,
)
