import numpy as np
import gymnasium as gym


class RewardShapingWrapper(gym.Wrapper):
    """
    Adds reward shaping to help with sparse rewards in MiniGrid Crossing.
    Works with the original MiniGrid environment (before any observation wrappers).
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.previous_distance_to_goal = None
        self.goal_position = None
        self.previous_agent_pos = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Find goal position directly from the unwrapped environment's grid
        grid = self.unwrapped.grid
        width, height = grid.width, grid.height
        
        # Find goal position (Goal object)
        self.goal_position = None
        for x in range(width):
            for y in range(height):
                cell = grid.get(x, y)
                if cell is not None and cell.type == 'goal':
                    self.goal_position = (x, y)
                    break
            if self.goal_position:
                break
        
        # Calculate initial distance to goal
        agent_pos = self.unwrapped.agent_pos
        self.previous_agent_pos = tuple(agent_pos)
        
        if self.goal_position is not None:
            self.previous_distance_to_goal = np.linalg.norm(
                np.array(agent_pos) - np.array(self.goal_position)
            )
        else:
            self.previous_distance_to_goal = 0
            
        return obs, info
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        current_agent_pos = tuple(self.unwrapped.agent_pos)
        shaped_reward = reward  # original reward, usually sparse

        if self.goal_position is not None:
            # Manhattan distance
            current_distance = abs(current_agent_pos[0] - self.goal_position[0]) + \
                            abs(current_agent_pos[1] - self.goal_position[1])

            # Normalize distance reward
            max_dist = abs(self.previous_agent_pos[0] - self.goal_position[0]) + \
                    abs(self.previous_agent_pos[1] - self.goal_position[1]) + 1e-8
            distance_reward = (self.previous_distance_to_goal - current_distance) / max_dist
            shaped_reward += distance_reward

            # Small penalty for each step
            shaped_reward -= 0.001

            # Reward for getting closer to goal (potential-based shaping)
            distance_reward = (self.previous_distance_to_goal - current_distance) * 0.1
            shaped_reward += distance_reward

            # Small reward for moving (action 2 is move forward)
            if action == 2 and current_agent_pos != self.previous_agent_pos:
                shaped_reward += 0.01
        
            # Penalty for stepping on lava
            cell = self.unwrapped.grid.get(*current_agent_pos)
            if cell is not None and cell.type == 'lava':
                shaped_reward -= 1.0  # Strong penalty
            
            if done and reward > 0:
                shaped_reward += 1.0
            
            
            # Update for next step
            self.previous_distance_to_goal = current_distance
        
        self.previous_agent_pos = current_agent_pos
        
        return obs, shaped_reward, done, truncated, info
