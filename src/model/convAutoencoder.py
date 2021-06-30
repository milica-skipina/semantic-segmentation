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
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, 7, stride=3),  # N, 84, 84
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 5, stride=3),  # N, 27, 27
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=2),  # N, 12, 12
            # nn.ReLU(),
            # nn.Conv2d(512, 256, 12),  # 256, 1, 1
        )
        # 256, 256
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, 256, 3, stride=1), # N, 254, 254
        #     nn.BatchNorm2d(num_features=256),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 512, 3, stride=1), # N, 252, 252
        #     nn.BatchNorm2d(num_features=512),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2), # 126, 126
        #     nn.Conv2d(512, 512, 3, stride=1),  # N, 124, 124
        #     nn.BatchNorm2d(num_features=512),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 256, 3, stride=1),  # 122, 122
        #     nn.BatchNorm2d(num_features=256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # 63, 63
        #     nn.Conv2d(256, 256, 3, stride=2),  # 31, 31
        #     nn.ReLU(),
        #     nn.BatchNorm2d(num_features=256),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 128, 3, stride=2), # N, 14, 14
        #     # nn.ReLU(),
        #     # nn.Conv2d(512, 256, 12), # 256, 1, 1
        # )
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 256, 3, stride=1), # N, 254, 254
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=1), # N, 252, 252
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) # 126, 126
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=1),  # N, 124, 124
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, stride=1),  # 122, 122
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),  # 63, 63
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, stride=2),  # 31, 31
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, stride=2), # N, 14, 14
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 512, 3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512),
            nn.ConvTranspose2d(512, 1024, 3, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=1024)
        )
        self.MaxUnpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 1024, 3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=1024),

        nn.ConvTranspose2d(1024, 512, 3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512)

        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            nn.ConvTranspose2d(256, 3, 3, stride=1),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(128, 256, 3, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(256, 256, 3, stride=2),
        #     nn.ReLU(),
        #     nn.MaxUnpool2d(kernel_size=2, stride=2)
            # nn.ConvTranspose2d(256, 512, 3, stride=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(512, 512, 3, stride=1),
            # nn.ReLU(),
            # # nn.MaxUnpool2d(kernel_size=2, stride=2),
            # # nn.ConvTranspose2d(512, 256, 3, stride=1),
            # # nn.ReLU(),
            # # nn.ConvTranspose2d(256, 3, 3, stride=1),
            # # nn.Sigmoid()

            # # nn.ConvTranspose2d(256, 512, 12), # N, 12, 12
            # # nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, stride=2), # N, 27, 27
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 5, stride=3, output_padding=1), # N, 84, 84
            nn.ReLU(),
            nn.ConvTranspose2d(256, 3, 7, stride=3), # N, 256, 256
            nn.Sigmoid()
        )

    def forward(self, x):
        # x, indices1 = self.enc1(x)
        # x, indices2 = self.enc2(x)
        # x = self.enc3(x)
        # x = self.dec1(x)
        # x = self.MaxUnpool(x, indices2)
        # x = self.dec2(x)
        # x = self.MaxUnpool(x, indices1)
        # x = self.dec3(x)
        # decoded = self.decoder(x, indices2)
        return self.decoder(self.encoder(x))
