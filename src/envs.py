import gym
from gym import spaces


class Envs(object):
    def __init__(self, name):
        self.name = name
        self.envs = gym.make(name).unwrapped
        self.action_n = self.envs.action_space.n
        self.action_type = self.envs.action_space.dtype
        self.obs_shape = self.envs.observation_space.shape
        self.obs_high = self.envs.observation_space.high
        self.obs_low = self.envs.observation_space.low
