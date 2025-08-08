

from dqn import create_dqn_model, get_policy_kwargs_cnn
from envs import Env, make_env, register_envs
from learning import curriculum_learning, fine_tune_from_checkpoint, fine_tune_from_checkpoints
from render_callback import checkpoint_callback
from test2 import init_model
from stable_baselines3.common.env_util import make_vec_env

from trained_models.constants import FINETUNED_DQN_5X5_CNN, BASELINE_DQN_7X7_CNN, BASELINE_MODEL_DIR


def finetune():
    fine_tune_from_checkpoint(
        checkpoint_path="trained_models/dqn_5x5_cnn_interval__80000_steps", 
        env_id=Env.Minigrid_7x7.value,
        output_path=f"{FINETUNED_DQN_5X5_CNN}_from_80000_steps",
        total_timesteps=120_000
        )


def train():
    env = make_vec_env(lambda: make_env(Env.Minigrid_7x7.value), n_envs=1)

    model = create_dqn_model(
        env, 
        "CnnPolicy", 
        get_policy_kwargs_cnn, 
        batch_size=512, 
        learning_rate=5e-4,
        exploration_fraction=0.8, 
        exploration_initial_eps=1
        )

    model.learn(total_timesteps=200_000, callback=checkpoint_callback)

    model.save(f"{BASELINE_MODEL_DIR}/{BASELINE_DQN_7X7_CNN}_160000_steps")

    # checkpoint_paths = [
    #     "trained_models/dqn_5x5_cnn_interval__40000_steps",
    #     "trained_models/dqn_5x5_cnn_interval__80000_steps",
    #     "trained_models/dqn_5x5_cnn_interval__120000_steps",
    #     # "trained_models/dqn_5x5_cnn_interval__160000_steps",
    #     # "trained_models/dqn_5x5_cnn_interval__200000_steps"
    # ]
    # fine_tune_from_checkpoints(checkpoint_paths, Env.Minigrid_7x7.value)


if __name__ == "__main__":
    register_envs()

    #train()

    finetune()