import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid.wrappers import FullyObsWrapper


class ObjectTypeAndAgentWrapper(gym.ObservationWrapper):
    """
    Custom wrapper that extracts:
    1. Only the object type channel from the full observation
    2. Agent position (x, y)
    3. Agent direction (0-3)
    
    This creates a flattened observation combining spatial object types with agent state.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # First apply FullyObsWrapper to get the full grid view
        if not isinstance(env, FullyObsWrapper):
            self.env = FullyObsWrapper(env)
        else:
            self.env = env
            
        # Get the original observation space from FullyObsWrapper
        original_obs_space = self.env.observation_space
        
        # Extract dimensions
        if isinstance(original_obs_space, spaces.Dict):
            image_space = original_obs_space['image']
            height, width, channels = image_space.shape
        else:
            height, width, channels = original_obs_space.shape
            
        # Calculate flattened size: height * width (object types) + 2 (position) + 1 (direction)
        flattened_size = height * width + 3
        
        # New observation space: flattened vector
        self.observation_space = spaces.Box(
            low=0,
            high=max(10, width, height),  # Max possible values
            shape=(flattened_size,),
            dtype=np.int32
        )
        
        """print(f"ObjectTypeAndAgentWrapper initialized:")
        print(f"  Grid size: {height}x{width}")
        print(f"  Object types size: {height * width}")
        print(f"  Agent info size: 3 (x, y, direction)")
        print(f"  Total observation size: {flattened_size}")"""
    
    def observation(self, obs):
        """
        Transform the observation to include only object types and agent info
        """
        # Handle both dict and array observations
        if isinstance(obs, dict):
            image = obs['image']
            direction = obs.get('direction', self.env.unwrapped.agent_dir)
        else:
            image = obs
            direction = self.env.unwrapped.agent_dir
            
        # Extract only the object type channel (channel 0)
        object_types = image[:, :, 0]  # Shape: (height, width)
        
        # Get agent position
        agent_pos = self.env.unwrapped.agent_pos
        agent_x, agent_y = agent_pos
        
        # Flatten object types and combine with agent info
        flattened_objects = object_types.flatten()  # Shape: (height * width,)
        agent_info = np.array([agent_x, agent_y, direction], dtype=np.int32)  # Shape: (3,)
        
        # Combine everything into one flattened observation
        combined_obs = np.concatenate([flattened_objects, agent_info])
        
        return combined_obs


class DirectionInImageWrapper(gym.ObservationWrapper):
    """
    Wrapper that modifies the image observation by replacing the agent's object type (10)
    with the agent's direction (0-3) at the agent's position in the object type layer.
    
    This allows the CNN to directly see the agent's direction encoded in the spatial image,
    rather than needing to handle direction as separate information.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # This wrapper should work with environments that provide image observations
        # It can be applied after FullyObsWrapper + ImgObsWrapper or just ImgObsWrapper
        
        # The observation space remains the same since we're just modifying values
        self.observation_space = env.observation_space
    
    def observation(self, obs):
        """
        Replace the agent's object type with its direction in the image observation
        """
        # Make a copy to avoid modifying the original observation
        modified_obs = obs.copy()
        
        # Get agent position and direction
        agent_pos = self.env.unwrapped.agent_pos
        agent_dir = self.env.unwrapped.agent_dir
        
        if agent_pos is not None:
            agent_x, agent_y = agent_pos
            
            # Ensure we're within bounds
            height, width = modified_obs.shape[:2]
            if 0 <= agent_x < width and 0 <= agent_y < height:
                # Replace the object type (channel 0) at agent position with direction + agent
                modified_obs[agent_y, agent_x, 0] += agent_dir
        
        return modified_obs
    

class DirectionalObsWrapper(FullyObsWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        obs = super().observation(obs)

        modified_obs = obs.copy()
        
        # Get agent position and direction
        agent_pos = self.unwrapped.agent_pos
        agent_dir = self.unwrapped.agent_dir

        x, y = agent_pos
        #print(agent_dir)
        modified_obs["image"][y, x, 0] = 10 + agent_dir  # encode direction in the object channel

        #print(modified_obs["image"].shape)

        return modified_obs
