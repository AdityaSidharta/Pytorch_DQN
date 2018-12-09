import torch

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")


class Config():
    def __init__(self, batch_size, gamma, eps_start, eps_end, eps_decay, target_update, device):
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.device = device

