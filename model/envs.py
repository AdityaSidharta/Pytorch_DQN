import gym

# TODO complete envs
class Envs(object):
    def __init__(self, name):
        self.name = name
        self.envs = gym.make(name).unwrapped
