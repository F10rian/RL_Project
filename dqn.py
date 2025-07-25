
from stable_baselines3 import DQN
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MiniGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[2]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).permute(0, 3, 1, 2).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.permute(0, 3, 1, 2)  # Convert to NCHW
        x = self.cnn(x)
        return self.linear(x)


def get_policy_kwargs():
    return dict(
        features_extractor_class=MiniGridCNN,
        features_extractor_kwargs=dict(features_dim=64),
    )

def get_dqn_model(vec_env, **kwargs):
    defaults = {
        "policy": "CnnPolicy",
        "env": vec_env,
        "verbose": 1,
        "learning_rate": 5e-4,
        "buffer_size": 50000,
        "learning_starts": 5000,
        "batch_size": 32,
        "gamma": 0.99,
        "train_freq": 4,
        "target_update_interval": 10000,
        "policy_kwargs": get_policy_kwargs(),
        "tensorboard_log": "./dqn_minigrid_tensorboard/",
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "exploration_fraction": 0.8
    }

    # Override defaults with any user-provided values
    config = {**defaults, **kwargs}

    model = DQN(**config)
    return model