"""
termination_wrapper.py
----------------------
"""

import gymnasium as gym
from gymnasium import Wrapper


class TerminationWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        env = self.env.unwrapped

        # Fast O(1) termination check using cached counter
        if (not env.dirt_comeback) and env._n_dirty == 0:
            terminated = True

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)