
from stable_baselines3.common.env_util import make_vec_env

from dqn import create_dqn_model_linear, get_policy_kwargs_cnn
from envs import Env, register_envs, make_env
from trained_models.constants import BASELINE_MODEL_DIR



def main():
    output_filename = "test_simple_crossing_02"
    env_id = Env.Minigrid_5x5
    register_envs()
    env = make_vec_env(lambda: make_env(env_id), n_envs=1)

    model = create_dqn_model_linear(
        env, 
        "CnnPolicy", 
        learning_rate=1e-5, 
        exploration_fraction=0.7,
        exploration_final_eps=0.1,
        batch_size=256,
        )
    model.learn(total_timesteps=500_000)

    model.save(f"{BASELINE_MODEL_DIR}/{output_filename}")



if __name__ == "__main__":
    main()