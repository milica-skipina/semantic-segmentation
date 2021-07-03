from pl_bolts.models.autoencoders import AE
import torch
from torch import nn
from torch.nn import functional as F


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(3, 256, 256, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 3, 256, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

