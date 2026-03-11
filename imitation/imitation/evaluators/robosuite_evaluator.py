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

    def evaluate(self, policy, epoch):
        print("\nEvaluating policy...")
        if self.save_video:
            current_video_folder = os.path.join(self.video_folder, f"epoch_{epoch}")
            os.makedirs(current_video_folder, exist_ok=True)

        success = np.zeros(self.n_rollouts)
        timsteps = np.full((self.n_rollouts,), self.max_steps)
        for n in tqdm(range(self.n_rollouts)):
            obs = self.env.reset()
            policy.reset()

            frames = []
            for steps in range(self.max_steps):
                if self.save_video:
                    frames.append(obs["agentview_image"].copy())

                action = policy.get_action(obs)
                if action.ndim > 1:
                    action = action.squeeze()
                obs, reward, done, _ = self.env.step(action)
                
                # self.env.render()
                if self.env.check_success():
                    success[n] = 1
                    timsteps[n] = steps
                    break
            
            if self.save_video:
                suffix = "success" if success[n] else "fail"
                video_path = os.path.join(current_video_folder, f"rollout_{n:03d}_{suffix}.mp4")
                writer = imageio.get_writer(video_path, fps=30)
                for frame in frames:
                    writer.append_data(frame.astype(np.uint8))
                writer.close()
        
        eval_info = {
            'success_rate': np.mean(success),
            'mean_timesteps': np.mean(timsteps),
        }

        print(f"Success rate: {eval_info['success_rate']}")
        print(f"Mean timesteps: {eval_info['mean_timesteps']}\n")

        return eval_info 