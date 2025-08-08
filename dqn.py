
import traceback
from stable_baselines3 import DQN
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper

from network import MiniGridCNN, MiniGridLinear
from sbx import DQN as SBX_DQN
from sbx.common.type_aliases import ReplayBufferSamplesNp


from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.preprocessing import is_image_space
from torch import nn

class DuelingCnnPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        kwargs['features_extractor_class'] = MiniGridCNN
        kwargs['features_extractor_kwargs'] = dict(features_dim=128)
        super().__init__(*args, **kwargs)


def get_policy_kwargs_cnn():
    return dict(
        features_extractor_class=MiniGridCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

def get_policy_kwargs_lin():
    return dict(
        features_extractor_class=MiniGridLinear,
        features_extractor_kwargs=dict(features_dim=128) 
    )


def create_dqn_model(vec_env, policy, policy_function, **kwargs):
    defaults = {
        "policy": policy,
        "env": vec_env,
        "verbose": 1,
        "learning_rate": 5e-4,
        "buffer_size": 100_000,
        "learning_starts": 1000,
        "batch_size": 64,
        "gamma": 0.99,
        "train_freq": 4,
        "policy_kwargs": policy_function(),
        "tensorboard_log": "./dqn_crossing_tensorboard/",
        "exploration_initial_eps": 1.0, # exploration rate the training starts with
        "exploration_final_eps": 0.1, # final exploration rate we reach
        "exploration_fraction": 0.8, # percent of training when we reach the final exploration rate
        "max_grad_norm": 0.5
    }

    # Override defaults with any user-provided values
    config = {**defaults, **kwargs}

    model = DQN(**config)
    return model


def create_sbx_model(vec_env, policy, policy_function, **kwargs):
    defaults = {
        "policy": policy,
        "env": vec_env,
        "verbose": 1,
        "learning_rate": 5e-4,
        "buffer_class":ReplayBufferSamplesNp,
        "buffer_size":100_000,
        "learning_starts": 1000,
        "batch_size": 64,
        "gamma": 0.99,
        "tau":1.0,
        "alpha":0.6,  # prioritization exponent
        "beta_start":0.4,  # initial IS correction
        "beta_end":1.0,    # final IS correction
        "beta_anneal_steps":100_000,
        "train_freq": 4,
        "policy_kwargs": policy_function(),
        "tensorboard_log": "./dqn_crossing_tensorboard/",
        "exploration_initial_eps": 1.0, # exploration rate the training starts with
        "exploration_final_eps": 0.1, # final exploration rate we reach
        "exploration_fraction": 0.8 # percent of training when we reach the final exploration rate
    }

    # Override defaults with any user-provided values
    config = {**defaults, **kwargs}

    model = SBX_DQN(**config)
    return model
