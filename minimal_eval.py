import os
import argparse
import uuid
from datetime import datetime
import robosuite as suite
from imitation.algo.base_algo import BaseAlgo
from imitation.evaluators.robosuite_evaluator import RobosuiteEvaluator
from imitation.utils.general_utils import AttrDict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="experiments/baseline_square/weights/weights_ep550.pth")
    parser.add_argument("--n_rollouts", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--output_dir", type=str, default="rollout_videos")
    args = parser.parse_args()

    # 1. Environment Config
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

    # 2. Evaluator Config – unique run dir to avoid overwrites
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:8]
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}_{run_id}")
    os.makedirs(args.output_dir, exist_ok=True)

    eval_config = AttrDict(
        env_config=env_config,
        n_rollouts=args.n_rollouts,
        max_steps=args.max_steps,
        save_video=True,
        save_npz=True,
        video_folder=run_dir,
    )

    # 3. Load Policy
    print(f"Loading checkpoint: {args.checkpoint}")
    policy = BaseAlgo.load_weights(args.checkpoint)
    policy.to("cpu")
    policy.eval()

    # 4. Run Evaluation
    evaluator = RobosuiteEvaluator(eval_config)
    eval_info = evaluator.evaluate(policy)

    print("\nEvaluation Results:", eval_info)
    print(f"Outputs saved to: {run_dir}")

if __name__ == "__main__":
    main()
