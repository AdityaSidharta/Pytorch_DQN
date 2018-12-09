class Agent(object):
    def __init__(self, learner, memory, policy, value_function, envs, config):
        self.learner = learner
        self.memory = memory
        self.policy = policy
        self.value_function = value_function
        self.envs = envs
        self.config = config

    def train_agent(self, n_iteration):
        pass

