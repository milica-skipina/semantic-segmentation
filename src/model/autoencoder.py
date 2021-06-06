import torch
from torch import nn
from torch.nn import functional as F


class AutoEncoder(nn.Module):
    def __init__(self, x, y, d):
        super().__init__()
        self.d = d
        self.encoder = nn.Sequential(
            nn.Linear(x*y*3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, x*y*3),
            nn.Tanh()
        )
        # self.encoder = nn.Sequential(
        #     nn.Linear(x * y * 3, d),
        #     nn.BatchNorm1d(d),
        #     nn.ReLU(),
        # )
        # self.output_layer_encoder = nn.Sequential(
        #     nn.Linear(d, d),
        #     nn.Tanh()
        # )
        #
        # self.layer1 = nn.Sequential(
        #     nn.ConvTranspose2d(d, 128, kernel_size=7, stride=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        # )
        #
        # self.layer2 = nn.Sequential(
        #     nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )
        #
        # self.layer3 = nn.Sequential(
        #     nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )
        #
        # self.layer4 = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        # )
        #
        # self.layer5 = nn.Sequential(
        #     nn.ConvTranspose2d(32, 32, kernel_size=2, stride=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
        #     nn.Tanh(),
        # )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # x = y.view(y.size()[0], -1)
        # x = self.output_layer_encoder(self.encoder(x))
        #
        # x = x.view(x.size()[0], self.d, 1, 1)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.layer5(x)
        return decoded
