model = DQN(
            "CnnPolicy",
            env,
            learning_rate=1e-4,  # Reduced learning rate for more stable learning
            buffer_size=50_000,  # Increased buffer size
            learning_starts=1000,  # Start learning after collecting more experience
            batch_size=64, #64,
            tau=1.0,
            gamma=0.99,
            train_freq=4,  # Train every 4 steps (more stable than every step)
            target_update_interval=1000,  # Update target network less frequently
            verbose=1,
            policy_kwargs=get_policy_kwargs_cnn(),
            tensorboard_log="./dqn_crossing_tensorboard/",
            exploration_initial_eps=0.8,  # Start with full exploration
            exploration_final_eps=0.1,   # End with 10% exploration (higher than default)
            exploration_fraction=0.6     # Explore for 30% of training (longer than default)
        )

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
    # print("Objects: ", object_channel[0, 0, :, :])  # Print the first channel of the first observation
    return self.linear(self.cnn(object_channel))
    
def transfer_feature_extractor(pretrained_model, model):
    """
    Transfer all feature extractor weights from pretrained model to new model.
    This will copy all compatible layers and report what was transferred.
    """
    # Get feature extractor state dicts
    pretrained_features_dict = pretrained_model.policy.q_net.features_extractor.state_dict()
    pretrained_features_dict_target = pretrained_model.policy.q_net_target.features_extractor.state_dict()
    
    # Load feature extractor weights into new model
    model.policy.q_net.features_extractor.load_state_dict(pretrained_features_dict)

    # Also update target network to match
    model.policy.q_net_target.features_extractor.load_state_dict(pretrained_features_dict_target)
    
    # Report results
    print(f"  - Transferred {len(pretrained_features_dict)} layers")
    for layer_name in list(pretrained_features_dict.keys())[:]:
        print(f"{layer_name}")
    
    return model