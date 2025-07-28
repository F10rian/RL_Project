import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MiniGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 3):
        super().__init__(observation_space, features_dim)

        print("Observation Space: ", observation_space)
        # Use only 1 channel (object type) instead of all 3 channels
        n_input_channels = 1  # Only object type channel

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass with single channel
        sample_obs = observation_space.sample()
        single_channel_sample = sample_obs[0:1, :, :]  # Only object type channel (first channel)
        # Already in (C, H, W) format for PyTorch
        
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(single_channel_sample[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Extract only the object type channel (channel 0)
        # Input shape: (batch, channels, height, width) or (channels, height, width)
        # We want: (batch, 1, height, width)
        
        if len(observations.shape) == 4:  # Batch of observations
            object_channel = observations[:, 0:1, :, :]  # Shape: (batch, 1, H, W)
        else:  # Single observation
            object_channel = observations[0:1, :, :].unsqueeze(0)  # Shape: (1, 1, H, W)
        print("Objects: ", object_channel[0, 0, :, :])  # Print the first channel of the first observation
        return self.linear(self.cnn(object_channel))


class MiniGridLinear(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 3):
        super().__init__(observation_space, features_dim)
        
        # Get input size from observation_space
        input_size = observation_space.shape[1]*observation_space.shape[2]
        #print(f'Observation Shape: {observation_space.shape}')
        #print(f'Observations Netword dim: {input_size}')
              
        self.linNet = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            # Remove final ReLU - let the policy head handle this
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        #self.print_minigrid(observations[0])
        #convert i.e. ([64, 3, 5, 5]) to ([64, 25])
        observations = observations[:, 0, :, :].view(observations.shape[0], -1)  # Flatten the input
        # Convert to float if needed (stable-baselines3 often passes int32)
        if observations.dtype != torch.float32:
            observations = observations.float()
        #print("Observations: ", observations.shape)
        self.print_minigrid(observations)
        
        # Observation is already flattened
        return self.linNet(observations)
    
    def print_minigrid(self, observations: torch.Tensor):
        print("Shape:", observations.shape)
        
        # Convert normalized values back to original integers for display
        original_values = (observations * 255).round().int()
        
        print("Original object types (channel 0):")
        print(original_values)
        # print("Original colors (channel 1):")
        # print(original_values[1])
        # print("Original states (channel 2):")
        # print(original_values[2])
        print("________________________")
        