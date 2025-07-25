
import traceback
from stable_baselines3 import DQN
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper

class MiniGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 3):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        print(f'Observation Shape: {observation_space.shape}')
        print(f'Feature Dim: {features_dim}')
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def get_policy_kwargs(env: ImgObsWrapper):
    return dict(
        features_extractor_class=MiniGridCNN,
        features_extractor_kwargs=dict(features_dim=env.action_space.n)
    )

def create_dqn_model(vec_env, **kwargs):
    defaults = {
        "policy": "MlpPolicy",
        "env": vec_env,
        "verbose": 1,
        "learning_rate": 1e-4,
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "batch_size": 64,
        "gamma": 0.99,
        "train_freq": 4,
        # "policy_kwargs": get_policy_kwargs(vec_env),
        "tensorboard_log": "./dqn_minigrid_tensorboard/",
        "exploration_initial_eps": 1.0, # exploration rate the training starts with
        "exploration_final_eps": 0.05, # final exploration rate we reach
        "exploration_fraction": 0.4 # percent of training when we reach the final exploration rate
    }

    # Override defaults with any user-provided values
    config = {**defaults, **kwargs}

    model = DQN(**config)
    return model