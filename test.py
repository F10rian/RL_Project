from envs import make_env, register_envs

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.vec_env import VecTransposeImage

from learning import finetune


register_envs()

env_id = "MiniGrid-Crossing-5x5-v0"

finetune("MiniGrid-Crossing-7x7-v0", "./dqn_best_model/best_model_pretrain_5x5")
