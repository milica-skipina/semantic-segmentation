import torch
from torch import nn
from torch.nn import functional as F


class AutoEncoder(nn.Module):
    def __init__(self, x, y, d):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(x * y, d),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d, x * y),
            nn.Tanh(),
        )

    def forward(self, y):
        h = self.encoder(y)
        y = self.decoder(h)
        return y
