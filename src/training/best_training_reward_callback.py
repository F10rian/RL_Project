import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class BestTrainingRewardCallback(BaseCallback):
    """
    Save the model when the mean training episode reward improves.
    This does NOT run extra evaluation episodes, only uses training data.
    """
    
    def __init__(self, save_path, save_freq=1000, window_size=10, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.window_size = window_size
        self.best_mean_reward = -np.inf
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Collect episode rewards from training
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        # Every save_freq steps, check mean reward and save if improved
        if self.n_calls % self.save_freq == 0 and len(self.episode_rewards) >= self.window_size:
            mean_reward = np.mean(self.episode_rewards[-self.window_size:])
            if self.verbose > 0:
                print(f"Step {self.n_calls}: Mean training reward (last {self.window_size} episodes): {mean_reward:.2f}")
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.save_path)
                if self.verbose > 0:
                    print(f"Saved new best training reward model: {self.save_path}.zip, mean reward: {mean_reward:.2f}")
        return True
