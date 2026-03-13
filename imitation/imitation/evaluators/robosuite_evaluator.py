import os
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

    def evaluate(self, policy, epoch=None):
        print("\nEvaluating policy...")
        if self.save_video or self.save_npz:
            if epoch is not None:
                current_folder = os.path.join(self.video_folder, f"epoch_{epoch}")
            else:
                current_folder = self.video_folder
            os.makedirs(current_folder, exist_ok=True)

        success = np.zeros(self.n_rollouts)
        timsteps = np.full((self.n_rollouts,), self.max_steps)
        for n in tqdm(range(self.n_rollouts)):
            obs = self.env.reset()
            policy.reset()

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

        print(f"Success rate: {eval_info['success_rate']}")
        print(f"Mean timesteps: {eval_info['mean_timesteps']}\n")

        return eval_info