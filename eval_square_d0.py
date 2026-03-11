"""
Standalone evaluation script for the square_d0 FlowPolicy checkpoint.

Usage:
    python eval_square_d0.py \
        --checkpoint weights_ep900.pth \
        --n_rollouts 20 \
        --max_steps 400 \
        --save_video \
        --video_dir rollout_videos \
        --device cuda

The checkpoint was saved by BaseAlgo.save_weights(), so it stores both the
config (including architecture kwargs and normalization stats) and the
state_dict. FlowPolicy.load_weights() (inherited from BaseAlgo) reconstructs
the full policy object without needing a separate config file.
"""

import os
import argparse
import numpy as np
import torch
import imageio
from tqdm import tqdm

import robosuite as suite
from imitation.algo.base_algo import BaseAlgo
from imitation.wrappers.robosuite_wrappers import RobosuiteImageFlipWrapper


# ---------------------------------------------------------------------------
# Environment configuration for SquareD0
# Mirrors the observations used during training (see flow_policy_config.py).
# ---------------------------------------------------------------------------
ENV_CONFIG = dict(
    env_name="NutAssemblySquare",
    robots="Panda",
    controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=["agentview", "robot0_eye_in_hand"],
    camera_heights=84,
    camera_widths=84,
    reward_shaping=False,
    horizon=400,
    ignore_done=False,
)


def build_env():
    """Create and wrap the robosuite Square environment."""
    env = suite.make(**ENV_CONFIG)
    env = RobosuiteImageFlipWrapper(env)  # flips camera images (training convention)
    return env


def collect_frame(env):
    """Grab an agentview RGB frame for video saving."""
    # robosuite renders into the observation dict; we re-use its last obs.
    # Caller is responsible for passing the current obs instead.
    raise RuntimeError("Use obs['agentview_image'] directly.")


def run_rollouts(policy, n_rollouts, max_steps, save_video, video_dir):
    """Run n_rollouts episodes and return success stats."""
    env = build_env()
    os.makedirs(video_dir, exist_ok=True)

    successes = []
    timesteps_list = []

    for ep in tqdm(range(n_rollouts), desc="Rollouts"):
        obs = env.reset()
        policy.reset()

        frames = []
        success = False
        ep_steps = max_steps

        for step in range(max_steps):
            if save_video:
                # agentview_image is already flipped by the wrapper (HxWxC)
                frames.append(obs["agentview_image"].copy())

            action = policy.get_action(obs)
            obs, _reward, done, _info = env.step(action)

            if env.check_success():
                success = True
                ep_steps = step + 1
                break

        successes.append(float(success))
        timesteps_list.append(ep_steps)

        if save_video:
            video_path = os.path.join(video_dir, f"rollout_{ep:03d}_{'success' if success else 'fail'}.mp4")
            writer = imageio.get_writer(video_path, fps=30)
            for frame in frames:
                writer.append_data(frame.astype(np.uint8))
            writer.close()
            print(f"  Episode {ep}: {'SUCCESS' if success else 'FAIL'} ({ep_steps} steps) -> {video_path}")
        else:
            print(f"  Episode {ep}: {'SUCCESS' if success else 'FAIL'} ({ep_steps} steps)")

    return np.array(successes), np.array(timesteps_list)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a FlowPolicy checkpoint on SquareD0.")
    parser.add_argument("--checkpoint", type=str, default="weights_ep900.pth",
                        help="Path to the .pth checkpoint (default: weights_ep900.pth)")
    parser.add_argument("--n_rollouts", type=int, default=20,
                        help="Number of evaluation rollouts (default: 20)")
    parser.add_argument("--max_steps", type=int, default=400,
                        help="Max steps per episode (default: 400)")
    parser.add_argument("--save_video", action="store_true",
                        help="Save per-episode MP4 rollout videos")
    parser.add_argument("--video_dir", type=str, default="rollout_videos",
                        help="Directory to save rollout videos (default: rollout_videos)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (default: cuda if available, else cpu)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load policy
    # ------------------------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    policy = BaseAlgo.load_weights(args.checkpoint)
    if policy is False:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    policy.to(args.device)
    policy.eval()
    print(f"Policy loaded on {args.device}: {type(policy).__name__}")

    # ------------------------------------------------------------------
    # 2. Run rollouts
    # ------------------------------------------------------------------
    print(f"\nRunning {args.n_rollouts} rollouts (max {args.max_steps} steps each)...")
    successes, timesteps = run_rollouts(
        policy=policy,
        n_rollouts=args.n_rollouts,
        max_steps=args.max_steps,
        save_video=args.save_video,
        video_dir=args.video_dir,
    )

    # ------------------------------------------------------------------
    # 3. Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print(f"Success rate:     {successes.mean():.1%}  ({int(successes.sum())}/{args.n_rollouts})")
    print(f"Mean timesteps:   {timesteps.mean():.1f}")
    print(f"Median timesteps: {np.median(timesteps):.1f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
