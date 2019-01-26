import torch


class Config:
    def __init__(
        self,
        batch_size,
        gamma,
        eps_start,
        eps_end,
        eps_decay,
        n_target_update,
        n_episode,
        device,
        memory_space,
        observation_space,
        action_space,
    ):
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.n_target_update = n_target_update
        self.n_episode = n_episode
        self.device = device
        self.memory_space = memory_space
        self.observation_space = observation_space
        self.action_space = action_space
