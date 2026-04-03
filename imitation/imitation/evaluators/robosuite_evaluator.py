import os
import random
import hashlib
import numpy as np
import imageio
from tqdm import tqdm
import robosuite as suite
from imitation.wrappers.robosuite_wrappers import RobosuiteImageFlipWrapper

class RobosuiteEvaluator:

    def __init__(self, eval_config, *args, **kwargs):
        # need to flip imgs in the environment
        self.env = RobosuiteImageFlipWrapper(suite.make(**eval_config.env_config))

        self.n_rollouts = eval_config.n_rollouts
        self.max_steps = eval_config.max_steps
        self.save_video = eval_config.get("save_video", False)
        self.video_folder = eval_config.get("video_folder", "rollout_videos")
        self.save_npz = eval_config.get("save_npz", False)

    def evaluate(self, policy, epoch=None, seed=None, verification_frames_dir=None):
        print("\nEvaluating policy...")
        if self.save_video or self.save_npz:
            if epoch is not None:
                current_folder = os.path.join(self.video_folder, f"epoch_{epoch}")
            else:
                current_folder = self.video_folder
            os.makedirs(current_folder, exist_ok=True)

        # Seed the RNG so the simulator starts in the same state for each
        # rollout index across different evaluate() calls with the same seed.
        # This works because robosuite's env.step() does not consume np.random
        # — only env.reset() does (for object placement sampling).
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        success = np.zeros(self.n_rollouts)
        timsteps = np.full((self.n_rollouts,), self.max_steps)
        init_states = []  # collect initial low-dim state for seed verification
        for n in tqdm(range(self.n_rollouts)):

            obs = self.env.reset()
            policy.reset()

            # Record initial state fingerprint for seed verification
            if seed is not None:
                init_obs = {}
                for key in sorted(obs.keys()):
                    if 'image' not in key:
                        init_obs[key] = np.array(obs[key], dtype=np.float64)
                init_states.append(init_obs)

            # Save first frame as a verification image so the user can visually
            # confirm that the nut/peg start in the same location across checkpoints.
            if verification_frames_dir is not None and n < 3:
                os.makedirs(verification_frames_dir, exist_ok=True)
                tag = f"ep{epoch}" if epoch is not None else "noepoch"
                img_path = os.path.join(
                    verification_frames_dir,
                    f"{tag}_rollout_{n:03d}_init.png"
                )
                imageio.imwrite(img_path, obs["agentview_image"].astype(np.uint8))
                print(f"  Saved verification frame: {img_path}")

            frames = []
            obs_history = {k: [] for k in obs.keys()}
            action_history = []

            for steps in range(self.max_steps):
                if self.save_video:
                    frames.append(obs["agentview_image"].copy())

                if self.save_npz:
                    for k, v in obs.items():
                        obs_history[k].append(np.array(v))

                action = policy.get_action(obs)
                if action.ndim > 1:
                    action = action.squeeze()

                if self.save_npz:
                    action_history.append(np.array(action))

                obs, reward, done, _ = self.env.step(action)
                
                if self.env.check_success():
                    success[n] = 1
                    timsteps[n] = steps
                    break

            suffix = "success" if success[n] else "fail"

            if self.save_video:
                video_path = os.path.join(current_folder, f"rollout_{n:03d}_{suffix}.mp4")
                writer = imageio.get_writer(video_path, fps=30, macro_block_size=1)
                for frame in frames:
                    writer.append_data(frame.astype(np.uint8))
                writer.close()

            if self.save_npz:
                npz_path = os.path.join(current_folder, f"rollout_{n:03d}_{suffix}.npz")
                obs_arrays = {f"obs_{k}": np.array(v) for k, v in obs_history.items()}
                np.savez(npz_path, actions=np.array(action_history), success=bool(success[n]), **obs_arrays)
        
        eval_info = {
            'success_rate': np.mean(success),
            'mean_timesteps': np.mean(timsteps),
        }

        # Print initial-state fingerprints so user can verify seeding works
        if seed is not None and init_states:
            print(f"\n--- Seed verification (seed={seed}) ---")
            # Print all low-dim obs keys for the first few rollouts
            for i in range(min(3, len(init_states))):
                print(f"  Rollout {i}:")
                for key, val in init_states[i].items():
                    print(f"    {key}: {val}")
            if len(init_states) > 3:
                print(f"  ... ({len(init_states) - 3} more rollouts omitted)")
            # Compute a single hash over all initial states for easy comparison
            all_vecs = [np.concatenate([v.ravel() for v in s.values()]) for s in init_states]
            state_bytes = np.concatenate(all_vecs).tobytes()
            digest = hashlib.md5(state_bytes).hexdigest()
            print(f"  Combined init-state MD5 digest: {digest}")
            print(f"--- End seed verification ---\n")
            eval_info['init_state_hash'] = digest

        print(f"Success rate: {eval_info['success_rate']}")
        print(f"Mean timesteps: {eval_info['mean_timesteps']}\n")

        return eval_info