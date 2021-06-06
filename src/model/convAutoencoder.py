# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class ConvAutoencoder(nn.Module):
#     def __init__(self):
#         super(ConvAutoencoder, self).__init__()
#
#         # Encoder
#         self.lin = nn.Linear(256*256*3, 16)
#         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
#         self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#
#         # Decoder
#         self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
#         self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)
#
#     def forward(self, x):
#         x = self.lin(x)
#         x = F.relu(x)
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = F.relu(self.t_conv1(x))
#         x = F.sigmoid(self.t_conv2(x))
#
#         return x

import torch
from torch import nn
from torch.nn import functional as F


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 256, 256
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, 7, stride=3), # N, 84, 84
            nn.ReLU(),
            nn.Conv2d(256, 512, 5, stride=3), # N, 27, 27
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=2), # N, 12, 12
            nn.ReLU(),
            nn.Conv2d(512, 256, 12), # 256, 1, 1
        )
        # 256, 1, 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 512, 12), # N, 12, 12
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, stride=2), # N, 27, 27
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 5, stride=3, output_padding=1), # N, 84, 84
            nn.ReLU(),
            nn.ConvTranspose2d(256, 3, 7, stride=3), # N, 256, 256
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
