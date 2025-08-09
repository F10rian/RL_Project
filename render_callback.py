from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from envs import make_env, register_envs

class RenderCallback(BaseCallback):
    def __init__(self, render_freq=1000, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self.training_env.render()
        return True


checkpoint_callback = CheckpointCallback(
    save_freq=40_000,               # Save every 40,000 steps
    save_path="./trained_models/", # Folder to save checkpoints
    name_prefix="dqn_5x5_cnn_interval_"         # File name prefix
)

def get_eval_callback(env_id, save_path="./"):
    eval_env = DummyVecEnv([lambda: make_env(env_id)])

    # Define callback for saving best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        eval_freq=5000,
        n_eval_episodes=20,
        deterministic=True,
        render=False
    )
    return eval_callback