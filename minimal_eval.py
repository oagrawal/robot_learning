import os
from datetime import datetime
import torch
import robosuite as suite
from imitation.algo.base_algo import BaseAlgo
from imitation.evaluators.robosuite_evaluator import RobosuiteEvaluator
from imitation.utils.general_utils import AttrDict

def main():
    # 1. Environment Config (from flow_policy_config.py and robosuite defaults)
    env_config = AttrDict(
        env_name="NutAssemblySquare",
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

    # 2. Evaluator Config (minimalist)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_video_dir = os.path.join("rollout_videos", f"run_{timestamp}")
    
    eval_config = AttrDict(
        env_config=env_config,
        n_rollouts=10,
        max_steps=400,
        save_video=True,
        video_folder=run_video_dir
    )

    # 3. Load Policy
    checkpoint_path = "weights_ep900.pth"
    print(f"Loading checkpoint: {checkpoint_path}")
    policy = BaseAlgo.load_weights(checkpoint_path)
    policy.to("cpu")
    policy.eval()

    # 4. Use Existing Evaluator
    evaluator = RobosuiteEvaluator(eval_config)
    eval_info = evaluator.evaluate(policy)

    print("\nEvaluation Results:", eval_info)

if __name__ == "__main__":
    main()
