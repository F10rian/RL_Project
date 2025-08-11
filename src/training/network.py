import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MiniGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        print("Observation Space: ", observation_space)
        # Use only 1 channel (object type) instead of all 3 channels
        n_input_channels = 1  # Only object type channel

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1),  # → (32, 3, 3)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),                # → (64, 2, 2)
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass with single channel
        sample_obs = observation_space.sample()
        single_channel_sample = sample_obs[0:1, :, :]  # Only object type channel (first channel)
        # Already in (C, H, W) format for PyTorch
        
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(single_channel_sample[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, features_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Extract only the object type channel (channel 0)
        # Input shape: (batch, channels, height, width) or (channels, height, width)
        # We want: (batch, 1, height, width)
        
        if len(observations.shape) == 4:  # Batch of observations
            object_channel = observations[:, 0:1, :, :]  # Shape: (batch, 1, H, W)
        else:  # Single observation
            object_channel = observations[0:1, :, :].unsqueeze(0)  # Shape: (1, 1, H, W)
        return self.linear(self.cnn(object_channel))
