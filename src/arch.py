from torch import nn as nn


class CartNet(nn.Module):
    def __init__(self, hidden_dim):
        super(CartNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.model_list = [
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 2),
        ]
        self.model = nn.Sequential(*self.model_list)

    def forward(self, x):
        return self.model(x)
