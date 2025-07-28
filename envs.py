import time
from minigrid.envs import CrossingEnv
from gymnasium.envs.registration import register

from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper, FullyObsWrapper, FlatObsWrapper
from custom_wrappers import DirectionalObsWrapper, ObjectTypeAndAgentWrapper, DirectionInImageWrapper
from reward_shaping import RewardShapingWrapper
import gymnasium as gym

import numpy as np


# Custom 4x4 version
class MiniGridCrossing5x5(CrossingEnv):
    def __init__(self, **kwargs):
        super().__init__(size=5, num_crossings=1, **kwargs)

class MiniGridCrossing7x7(CrossingEnv):
    def __init__(self, **kwargs):
        super().__init__(size=7, num_crossings=1, **kwargs)

class MiniGridCrossing11x11(CrossingEnv):
    def __init__(self, **kwargs):
        super().__init__(size=11, num_crossings=1, **kwargs)

def register_envs():
    # Register them to gym
    register(
        id='MiniGrid-Crossing-5x5-v0',
        entry_point='envs:MiniGridCrossing5x5',
    )

    register(
        id='MiniGrid-Crossing-7x7-v0',
        entry_point='envs:MiniGridCrossing7x7',
    )

    register(
        id='MiniGrid-Crossing-11x11-v0',
        entry_point='envs:MiniGridCrossing11x11',
    )

def make_env(env_id):
    env = gym.make(env_id)#, render_mode="human")
    #env = RewardShapingWrapper(env)  # Enable reward shaping for better learning
    env = DirectionalObsWrapper(env)  # This gives you the full 5x5 grid view
    env = ImgObsWrapper(env)    # This extracts just the image from the dict
    return env


def test_and_visualize_envs(env_id):
    env = make_env(env_id)
    env.reset()
    
    while True:
        env.reset()
        env.render()
        time.sleep(1)

def test_and_visualize_env_with_random_walk(env_id):
    env = make_env(env_id)
    obs = env.reset()
    
    while True:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        time.sleep(0.2)
        if done:
            break

    env.close()