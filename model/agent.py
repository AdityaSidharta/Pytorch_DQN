class Agent(object):
    def __init__(self, memory, q_function, policy, config, envs):
        self.memory = memory
        self.q_function = q_function
        self.policy = policy
        self.config = config
        self.envs = envs
