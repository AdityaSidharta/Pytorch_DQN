import random
import numpy as np


# State, Action, Reward, Next State
class Memory(object):
    def __init__(self, capacity, n_state):
        self.capacity = capacity
        self.memory = list()
        self.position = 0
        self.n_state = n_state

    def __len__(self):
        return len(self.memory)

    def is_memory_full(self):
        return len(self.memory) == self.capacity

    def update_position(self):
        self.position = self.position + 1
        if self.position == self.capacity:
            self.position = 0

    def save(self, state, action, reward, next_state, finish):
        assert len(state) == self.n_state
        if finish:
            next_state = [np.nan] * self.n_state
        if self.is_memory_full():
            self.memory[self.position] = (state, action, reward, next_state, finish)
        else:
            self.memory.extend([state, action, reward, next_state, finish])
        self.update_position()

    def sample(self, sample_size, torch=False, device=None):
        if sample_size > len(self.memory):
            raise ValueError("Sample size is bigger than the memory size")
        else:
            sample_list = random.sample(self.memory, sample_size)
            state = np.stack([x[0] for x in sample_list])
            action = np.stack([x[1] for x in sample_list])
            reward = np.stack([x[2] for x in sample_list])
            next_state = np.stack([x[3] for x in sample_list])
            finish = np.stack([x[4] for x in sample_list])
        if torch:
            assert device is not None
            state_tensor = torch.from_numpy(state).to(device)
            action_tensor = torch.from_numpy(action).to(device)
            reward_tensor = torch.from_numpy(reward).to(device)
            next_state_tensor = torch.from_numpy(next_state).to(device)
            finish_tensor = torch.from_numpy(finish).to(device)
            return (
                state_tensor,
                action_tensor,
                reward_tensor,
                next_state_tensor,
                finish_tensor,
            )
        else:
            return state, action, reward, next_state, finish
