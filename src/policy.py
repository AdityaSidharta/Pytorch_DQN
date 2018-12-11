import numpy as np
import math
import random


class Policy:
    def select_action(self, value_function, config):
        raise NotImplementedError()

    def best_action(self, value_function, config):
        raise NotImplementedError()


class EGreedy(Policy):
    def __init__(self):
        self.n_actions = 0

    def calc_eps_threshold(self, config):
        return config.eps_end + (config.eps_start - config.eps_end) * (
            math.exp(-1.0 * self.n_actions / config.eps_decay)
        )

    def update_n_actions(self):
        self.n_actions = self.n_actions + 1

    def reset_n_actions(self):
        self.n_actions = 0

    def select_action(self, value_function, config):
        eps_threshold = self.calc_eps_threshold(config)
        self.update_n_actions()
        random_number = random.random()

        if random_number > eps_threshold:
            return np.argmax(value_function)
        else:
            return random.randrange(len(value_function))

    def best_action(self, value_function, config):
        return np.argmax(value_function)
