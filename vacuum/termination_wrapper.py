# file: termination_wrapper.py
"""
Define a wrapper of VacuumCleanerWorldEnv (vacuum.world) to terminate an episode 
after all rooms are cleaned under the assumption that dirt won't comeback, 
that is, dirt_comeback should be set to False
"""
import gymnasium as gym
from gymnasium import Wrapper

class TerminationWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        assert not env.unwrapped.dirt_comeback, \
            "[AssertionViolation] cannot terminate when dirt may comeback!"

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.env.unwrapped._n_dirty == 0:
            terminated = True
            info['terminated_by_wrapper'] = True
        return obs, reward, terminated, truncated, info