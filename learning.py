import zipfile
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from dqn import get_dqn_model, get_policy_kwargs
from envs import make_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env



def train(env_id, model, env):
    eval_env = DummyVecEnv([lambda: make_env(env_id)])

    # Define callback for saving best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./dqn_best_model/",
        log_path="./dqn_eval_logs/",
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    model.learn(total_timesteps=100_000, callback=eval_callback)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

def pretrain(env_id):
    vec_env = make_vec_env(lambda: make_env(env_id), n_envs=1)
    vec_env = VecTransposeImage(vec_env)

    model = get_dqn_model(vec_env)

    train(env_id, model, vec_env)


def finetune(env_id, model_path):
    vec_env = make_vec_env(lambda: make_env(env_id), n_envs=1)
    vec_env = VecTransposeImage(vec_env)
    
    model = DQN.load(model_path, env=vec_env)

    new_model = get_dqn_model(
        vec_env,
        exploration_initial_eps=0.8,
        exploration_final_eps=0.05,
        exploration_fraction=0.7
    )
    new_model.policy.load_state_dict(model.policy.state_dict())

    train(env_id, new_model, vec_env)