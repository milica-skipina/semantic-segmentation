# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
#
# from src.data_loader.cityscapes_dataset import CityscapesDataset
# from src.model.myAutoEncoder import Autoencoder
#
#
# def train(model, num_epochs=5, batch_size=2, learning_rate=1e-3):
#     torch.manual_seed(42)
#     criterion = nn.MSELoss() # mean square error loss
#     optimizer = torch.optim.Adam(model.parameters(),
#                                  lr=learning_rate,
#                                  weight_decay=1e-5) # <--
#     train_loader = torch.utils.data.DataLoader(mnist_data,
#                                                batch_size=batch_size,
#                                                shuffle=True)
#     train_loader = DataLoader(CityscapesDataset('../../data/raw/leftImg8bit/train', '../../data/raw/gtFine/train'),
#                               batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False
#                               )
#
#
#     outputs = []
#     for epoch in range(num_epochs):
#         for data in train_loader:
#             img, _ = data
#             recon = model(img)
#             loss = criterion(recon, img)
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#
#         print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
#         outputs.append((epoch, img, recon),)
#     return outputs
#
# mnist_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
# mnist_data = list(mnist_data)[:4096]
#
# model = Autoencoder()
# max_epochs = 2000
# outputs = train(model, num_epochs=max_epochs)
#
# for k in range(0, max_epochs, 20):
#     plt.figure(figsize=(9, 2))
#     imgs = outputs[k][1].detach().numpy()
#     recon = outputs[k][2].detach().numpy()
#     for i, item in enumerate(imgs):
#         if i >= 9: break
#         plt.subplot(2, 9, i + 1)
#         plt.imshow(item[0])
#
#     for i, item in enumerate(recon):
#         if i >= 9: break
#         plt.subplot(2, 9, 9 + i + 1)
#         plt.imshow(item[0])
#
#     plt.show()

# ae = AE(input_height=256)
# print(AE.pretrained_weights_available())
# ae = ae.from_pretrained('cifar10-resnet18')
# ae.freeze()
# ae = nn.DataParallel(ae)
# ae.to(DEVICE)

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

from src.main import prepare_data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, val_loader = prepare_data()

model = models.resnet18(pretrained=True)
for params in model.parameters():
    params.requires_grad = False
num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 2)
model.to(DEVICE)