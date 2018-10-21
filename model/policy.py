import numpy as np
import math
import random


class EGreedy(object):
    def __init__(self, config):
        self.gamma = config["GAMMA"]
        self.eps_start = config["EPS_START"]
        self.eps_end = config["EPS_END"]
        self.eps_decay = config["EPS_DECAY"]
        self.n_actions = 0

    def calc_eps_threshold(self):
        return self.eps_end + (self.eps_start - self.eps_end) * (
            math.exp(-1. * self.n_actions / self.eps_decay)
        )

    def update_n_actions(self):
        self.n_actions = self.n_actions + 1

    def select_action(self, value_function):
        eps_threshold = self.calc_eps_threshold()
        self.update_n_actions()
        random_number = random.random()

        if random_number > eps_threshold:
            return np.argmax(value_function)
        else:
            return random.randrange(len(value_function))
