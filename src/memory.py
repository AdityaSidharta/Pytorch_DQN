import random
import numpy as np
import torch


# State, Action, Reward, Next State
class Memory(object):
    def __init__(self, capacity, n_state):
        self.capacity = capacity
        self.n_state = n_state
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def is_memory_full(self):
        return len(self.memory) == self.capacity

    def update_position(self):
        self.position = self.position + 1
        if self.position == self.capacity:
            self.position = 0

    def save(
        self, state: np.array, action: int, reward: float, next_state: np.array, finish: int
    ):
        state, next_state = state.tolist(), next_state.tolist()
        assert len(state) == self.n_state
        next_state = [np.nan] * self.n_state if finish else next_state
        if self.is_memory_full():
            self.memory[self.position] = (state, action, reward, next_state, finish)
        else:
            self.memory.append([state, action, reward, next_state, finish])
        self.update_position()

    def sample(self, sample_size, return_tensor=False, torch_device=None):
        if torch_device is None:
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if sample_size > len(self.memory):
            raise ValueError("Sample size is bigger than the memory size")
        else:
            sample_list = random.sample(self.memory, sample_size)
            state = np.stack([x[0] for x in sample_list]).astype(float)
            action = np.stack([x[1] for x in sample_list]).astype(int)
            reward = np.stack([x[2] for x in sample_list]).astype(float)
            next_state = np.stack([x[3] for x in sample_list]).astype(float)
            finish = np.stack([x[4] for x in sample_list]).astype(int)
        if return_tensor:
            state_tensor = torch.from_numpy(state).to(torch_device, dtype=torch.float)
            action_tensor = torch.from_numpy(action).to(torch_device, dtype=torch.long)
            reward_tensor = torch.from_numpy(reward).to(torch_device, dtype=torch.float)
            next_state_tensor = torch.from_numpy(next_state).to(
                torch_device, dtype=torch.float
            )
            finish_tensor = torch.from_numpy(finish).to(torch_device, dtype=torch.long)
            return (
                state_tensor,
                action_tensor,
                reward_tensor,
                next_state_tensor,
                finish_tensor,
            )
        else:
            return state, action, reward, next_state, finish
