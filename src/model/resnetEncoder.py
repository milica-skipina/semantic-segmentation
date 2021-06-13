import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import torch
from torch import nn
from torch.nn import functional as F

from pl_bolts.models.autoencoders import AE
from torch.autograd import Variable

class MyResnetEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        for param in resnet.parameters():
            param.requires_grad_ = False
        # self.encoder = nn.Sequential(*modules)

        # resnet = models.resnet152(pretrained=True)
        # modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        # self.fc1 = nn.Linear(resnet.fc.in_features, 1024)
        # self.bn1 = nn.BatchNorm1d(1024, momentum=0.01)
        # self.fc2 = nn.Linear(1024, 2048)
        # self.bn2 = nn.BatchNorm1d(2048, momentum=0.01)
        self.fc3 = nn.Linear(resnet.fc.in_features, 256)
        self.bn3 = nn.BatchNorm1d(256, momentum=0.01)
        # Latent vectors mu and sigma
        self.relu = nn.ReLU(inplace=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 512, 12, stride=2, output_padding=1),  # N, 12, 12
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, stride=2),  # N, 27, 27
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 5, stride=3, output_padding=1),  # N, 84, 84
            nn.ReLU(),
            nn.ConvTranspose2d(256, 3, 7, stride=3),  # N, 256, 256
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        # x = self.bn1(self.fc1(x))
        # x = self.relu(x)
        # x = self.bn2(self.fc2(x))
        # x = self.relu(x)
        x = self.bn3(self.fc3(x))
        x = self.relu(x)

        # x = x.view(x.size(0), -1)  # flatten output of conv
        shape = (-1, 256, 1, 1)
        x = x.view(x.size(0), *shape[1:])  # 64*16*16 -> 64,16,16
        x = self.decoder(x)
        return x

# from src.main import prepare_data
#
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# train_loader, val_loader = prepare_data()
#
# model = models.resnet18(pretrained=True)
# for params in model.parameters():
#     params.requires_grad = False
# num_ftrs = model.fc.in_features
#
# model.fc = nn.Linear(num_ftrs, 2)
# model.to(DEVICE)