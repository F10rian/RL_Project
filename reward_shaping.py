import numpy as np
import gymnasium as gym
from custom_wrappers import ObjectTypeAndAgentWrapper
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper, FullyObsWrapper, FlatObsWrapper


class RewardShapingWrapper(gym.Wrapper):
    """
    Adds reward shaping to help with sparse rewards in MiniGrid Crossing
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.previous_distance_to_goal = None
        self.goal_position = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Find goal position in the grid
        # For ObjectTypeAndAgentWrapper, we need to reconstruct the grid
        if hasattr(self.env, 'unwrapped'):
            # Get grid size from observation space
            grid_size = int(np.sqrt(len(obs) - 3))  # -3 for agent info
            grid = obs[:-3].reshape(grid_size, grid_size)
            
            # Find goal (object type 8)
            goal_positions = np.where(grid == 8)
            if len(goal_positions[0]) > 0:
                self.goal_position = (goal_positions[0][0], goal_positions[1][0])
            else:
                self.goal_position = None
        
        # Calculate initial distance
        agent_pos = obs[-3:-1]  # x, y position from observation
        if self.goal_position is not None:
            self.previous_distance_to_goal = np.sqrt(
                (agent_pos[0] - self.goal_position[0])**2 + 
                (agent_pos[1] - self.goal_position[1])**2
            )
        else:
            self.previous_distance_to_goal = 0
            
        return obs, info
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Add distance-based reward shaping
        if self.goal_position is not None:
            agent_pos = obs[-3:-1]  # x, y position from observation
            current_distance = np.sqrt(
                (agent_pos[0] - self.goal_position[0])**2 + 
                (agent_pos[1] - self.goal_position[1])**2
            )
            
            # Reward for getting closer to goal
            distance_reward = (self.previous_distance_to_goal - current_distance) * 0.1
            
            # Small positive reward for moving (to encourage exploration)
            movement_reward = 0.01 if action == 2 else 0  # action 2 is forward
            
            # Penalty for standing still
            stillness_penalty = -0.05 if action in [3, 4, 5, 6] else 0 #-0.005
            
            shaped_reward = reward + distance_reward + movement_reward + stillness_penalty
            
            self.previous_distance_to_goal = current_distance
        else:
            shaped_reward = reward
            
        return obs, shaped_reward, done, truncated, info


def make_env_with_reward_shaping(env_id):
    """Create environment with reward shaping"""
    env = gym.make(env_id)
    env = FullyObsWrapper(env)
    #env = ObjectTypeAndAgentWrapper(env)
    env = RewardShapingWrapper(env)
    return env
