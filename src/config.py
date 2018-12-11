import torch

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
N_TARGET_UPDATE = 10
N_EPISODE = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    ):
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.n_target_update = n_target_update
        self.n_episode = n_episode
        self.device = device
