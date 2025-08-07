

from dqn import create_dqn_model, get_policy_kwargs_cnn
from envs import Env, make_env, register_envs
from learning import curriculum_learning, fine_tune_from_checkpoints
from render_callback import checkpoint_callback
from test2 import init_model
from stable_baselines3.common.env_util import make_vec_env


def main():
    register_envs()

    # env = make_vec_env(lambda: make_env(Env.Minigrid_5x5.value), n_envs=1)

    # model = create_dqn_model(env, "CnnPolicy", get_policy_kwargs_cnn, batch_size=512)

    # model.learn(total_timesteps=200_000, callback=checkpoint_callback)


    checkpoint_paths = [
        "trained_models/dqn_5x5_cnn_interval__40000_steps",
        "trained_models/dqn_5x5_cnn_interval__80000_steps",
        "trained_models/dqn_5x5_cnn_interval__120000_steps",
        # "trained_models/dqn_5x5_cnn_interval__160000_steps",
        # "trained_models/dqn_5x5_cnn_interval__200000_steps"
    ]
    fine_tune_from_checkpoints(checkpoint_paths, Env.Minigrid_7x7.value)
    



if __name__ == "__main__":
    main()