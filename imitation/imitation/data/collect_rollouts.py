"""
Collect rollouts from a trained policy checkpoint with dense shaped rewards.
Saves successes and failures into separate HDF5 files in a single loop.
Also saves the first 20 videos of each outcome for visual verification.

Usage:
    python collect_rollouts.py \
        --checkpoint ~/robot_learning/experiments/my_run/weights/weights_ep500.pth \
        --output_dir ./ \
        --n_successes 500 \
        --n_failures 500 \
        --max_steps 400
"""

import os
import argparse
import numpy as np
import h5py
import imageio
from tqdm import tqdm

import robosuite as suite
from imitation.algo.base_algo import BaseAlgo
from imitation.wrappers.robosuite_wrappers import RobosuiteImageFlipWrapper


def collect_rollouts(checkpoint_path, output_dir, n_successes, n_failures, max_steps, n_videos=20):
    # Load policy
    print(f"Loading policy from {checkpoint_path}")
    policy = BaseAlgo.load_weights(checkpoint_path)
    policy.to('cuda')
    policy.eval()

    # Create env with reward_shaping=True for dense rewards
    env = RobosuiteImageFlipWrapper(suite.make(
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
    ))

    os.makedirs(output_dir, exist_ok=True)
    success_path = os.path.join(output_dir, 'success.hdf5')
    failure_path = os.path.join(output_dir, 'failure_dense.hdf5')
    video_dir = os.path.join(output_dir, 'collection_videos')

    f_succ, succ_grp = None, None
    f_fail, fail_grp = None, None
    if n_successes > 0:
        f_succ = h5py.File(success_path, 'w')
        succ_grp = f_succ.create_group('data')
    if n_failures > 0:
        f_fail = h5py.File(failure_path, 'w')
        fail_grp = f_fail.create_group('data')

    success_count = 0
    failure_count = 0
    total_rollouts = 0

    total_needed = n_successes + n_failures
    pbar = tqdm(total=total_needed, desc=f"Collecting (S:0/{n_successes} F:0/{n_failures})")

    while success_count < n_successes or failure_count < n_failures:
        obs = env.reset()
        policy.reset()

        obs_history = {k: [] for k in obs.keys()}
        action_history = []
        reward_history = []
        shaped_reward_history = []
        frames = []

        need_succ_video = success_count < n_videos
        need_fail_video = failure_count < n_videos

        success = False
        for step in range(max_steps):
            for k, v in obs.items():
                obs_history[k].append(np.array(v))

            # Save frames if we might need a video
            if need_succ_video or need_fail_video:
                frames.append(obs["agentview_image"].copy())

            action = policy.get_action(obs)
            if action.ndim > 1:
                action = action.squeeze()
            action_history.append(np.array(action))

            obs, reward, done, info = env.step(action)
            reward_history.append(reward)

            try:
                staged = env.env.staged_rewards()
                shaped_reward_history.append(np.array(staged))
            except:
                shaped_reward_history.append(np.array([reward, 0.0, 0.0, 0.0]))

            if env.check_success():
                success = True
                break

        total_rollouts += 1

        if success and success_count < n_successes and succ_grp is not None:
            demo_key = f"demo_{success_count}"
            demo_grp = succ_grp.create_group(demo_key)
            demo_grp.create_dataset('actions', data=np.array(action_history, dtype=np.float32))
            demo_grp.create_dataset('rewards', data=np.array(reward_history, dtype=np.float32))
            dones = np.zeros(len(action_history), dtype=np.float32)
            dones[-1] = 1.0
            demo_grp.create_dataset('dones', data=dones)
            obs_grp = demo_grp.create_group('obs')
            for k, v_list in obs_history.items():
                obs_grp.create_dataset(k, data=np.array(v_list))

            success_count += 1
            pbar.update(1)

            # Save video
            if success_count <= n_videos and frames:
                vid_dir = os.path.join(video_dir, 'successes')
                os.makedirs(vid_dir, exist_ok=True)
                writer = imageio.get_writer(os.path.join(vid_dir, f'success_{success_count:03d}.mp4'), fps=30, macro_block_size=1)
                for frame in frames:
                    writer.append_data(frame.astype(np.uint8))
                writer.close()

        elif not success and failure_count < n_failures and fail_grp is not None:
            demo_key = f"demo_{failure_count}"
            demo_grp = fail_grp.create_group(demo_key)
            demo_grp.create_dataset('actions', data=np.array(action_history, dtype=np.float32))
            demo_grp.create_dataset('rewards', data=np.array(reward_history, dtype=np.float32))
            demo_grp.create_dataset('staged_rewards', data=np.array(shaped_reward_history, dtype=np.float32))
            dones = np.zeros(len(action_history), dtype=np.float32)
            dones[-1] = 1.0
            demo_grp.create_dataset('dones', data=dones)
            obs_grp = demo_grp.create_group('obs')
            for k, v_list in obs_history.items():
                obs_grp.create_dataset(k, data=np.array(v_list))

            failure_count += 1
            pbar.update(1)

            # Save video
            if failure_count <= n_videos and frames:
                vid_dir = os.path.join(video_dir, 'failures')
                os.makedirs(vid_dir, exist_ok=True)
                writer = imageio.get_writer(os.path.join(vid_dir, f'failure_{failure_count:03d}.mp4'), fps=30, macro_block_size=1)
                for frame in frames:
                    writer.append_data(frame.astype(np.uint8))
                writer.close()

        # Update progress bar description
        pbar.set_description(f"Collecting (S:{success_count}/{n_successes} F:{failure_count}/{n_failures})")

    pbar.close()
    if f_succ is not None:
        f_succ.close()
    if f_fail is not None:
        f_fail.close()

    print(f"\nDone! {total_rollouts} total rollouts")
    if n_successes > 0:
        print(f"  Successes: {success_count} → {success_path}")
    if n_failures > 0:
        print(f"  Failures:  {failure_count} → {failure_path}")
    print(f"  Videos:    {video_dir}/")
    print(f"  Policy success rate: {success_count/total_rollouts:.1%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to policy checkpoint')
    parser.add_argument('--output_dir', type=str, default='./', help='Output directory')
    parser.add_argument('--n_successes', type=int, default=500, help='Number of successful rollouts')
    parser.add_argument('--n_failures', type=int, default=500, help='Number of failed rollouts')
    parser.add_argument('--max_steps', type=int, default=400, help='Max steps per rollout')
    parser.add_argument('--n_videos', type=int, default=20, help='Number of videos to save per outcome')
    args = parser.parse_args()

    collect_rollouts(args.checkpoint, args.output_dir, args.n_successes, args.n_failures, args.max_steps, args.n_videos)
