"""
Collect failure rollouts from a trained flow policy checkpoint with dense shaped rewards.
Saves per-step shaped rewards (reaching, grasping, lifting, hovering) for fine-grained 
timestep labeling — used in CFG to label which timesteps are positive vs negative.

Usage:
    python collect_failures.py \
        --checkpoint ~/robot_learning/experiments/my_run/weights/weights_ep500.pth \
        --output ./data/failure_dense.hdf5 \
        --n_rollouts 500 \
        --max_steps 400
"""

import os
import argparse
import numpy as np
import h5py
from tqdm import tqdm

import robosuite as suite
from imitation.algo.base_algo import BaseAlgo
from imitation.wrappers.robosuite_wrappers import RobosuiteImageFlipWrapper


def collect_failures(checkpoint_path, output_path, n_rollouts, max_steps, n_target_failures=None):
    """
    Run rollouts with a trained policy and save ONLY the failed episodes.
    Uses reward_shaping=True to get dense per-step rewards.
    """

    # Load policy from checkpoint
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
        reward_shaping=True,   # <-- DENSE rewards
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=84,
        camera_widths=84,
    ))

    target = n_target_failures if n_target_failures else n_rollouts
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        data_grp = f.create_group('data')
        
        failure_count = 0
        total_rollouts = 0
        
        pbar = tqdm(total=target, desc="Collecting failures")
        
        while failure_count < target:
            obs = env.reset()
            policy.reset()
            
            # Collect trajectory data
            obs_history = {k: [] for k in obs.keys()}
            action_history = []
            reward_history = []
            shaped_reward_history = []  # raw shaped reward per step
            
            success = False
            for step in range(max_steps):
                # Store obs
                for k, v in obs.items():
                    obs_history[k].append(np.array(v))
                
                # Get action
                action = policy.get_action(obs)
                if action.ndim > 1:
                    action = action.squeeze()
                action_history.append(np.array(action))
                
                # Step env — reward_shaping=True gives dense reward
                obs, reward, done, info = env.step(action)
                reward_history.append(reward)
                
                # Also get the 4 staged reward components for detailed analysis
                try:
                    staged = env.env.staged_rewards()
                    shaped_reward_history.append(np.array(staged))  # (reaching, grasping, lifting, hovering)
                except:
                    shaped_reward_history.append(np.array([reward, 0.0, 0.0, 0.0]))
                
                if env.check_success():
                    success = True
                    break
            
            total_rollouts += 1
            
            # Only save failures
            if not success:
                demo_key = f"demo_{failure_count}"
                demo_grp = data_grp.create_group(demo_key)
                
                # Save actions
                demo_grp.create_dataset('actions', data=np.array(action_history, dtype=np.float32))
                
                # Save rewards (dense shaped)
                demo_grp.create_dataset('rewards', data=np.array(reward_history, dtype=np.float32))
                
                # Save staged rewards (4 components) for analysis
                demo_grp.create_dataset('staged_rewards', data=np.array(shaped_reward_history, dtype=np.float32))
                
                # Save dones
                dones = np.zeros(len(action_history), dtype=np.float32)
                dones[-1] = 1.0
                demo_grp.create_dataset('dones', data=dones)
                
                # Save observations
                obs_grp = demo_grp.create_group('obs')
                for k, v_list in obs_history.items():
                    obs_grp.create_dataset(k, data=np.array(v_list))
                
                failure_count += 1
                pbar.update(1)
        
        pbar.close()
        print(f"\nDone! Collected {failure_count} failures from {total_rollouts} total rollouts.")
        print(f"Failure rate: {failure_count/total_rollouts:.1%}")
        print(f"Saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to policy checkpoint (weights_epN.pth)')
    parser.add_argument('--output', type=str, default='./data/failure_dense.hdf5', help='Output HDF5 path')
    parser.add_argument('--n_rollouts', type=int, default=500, help='Number of failure rollouts to collect')
    parser.add_argument('--max_steps', type=int, default=400, help='Max steps per rollout')
    args = parser.parse_args()

    collect_failures(args.checkpoint, args.output, args.n_rollouts, args.max_steps)
