import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class SemanticSegmentationModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        # 256, 256
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad_ = False

        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(256, 512, 12), # N, 12, 12
            # nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, stride=2), # N, 27, 27
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 784, 1, stride=1),
            nn.BatchNorm2d(num_features=784),
            nn.ReLU(),
            nn.ConvTranspose2d(784, 1024, 5, stride=3, output_padding=1), # N, 84, 84
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 8, 7, stride=3), # N, 256, 256
            # nn.Softmax()
        )

        # self.fc = nn.Sequential(
        #     nn.Linear(256*256*3, 4),
        #     nn.Softmax(),
        #     nn.Linear(4, 256*256*3)
        # )

    def forward(self, x):
        encoded = self.encoder(x)
        x = self.decoder(encoded)
        # x = torch.argmax(x, dim=1)
        # x = self.fc(nn.Flatten(decoded))

        return x
