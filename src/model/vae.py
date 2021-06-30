import torch
import torch.nn as nn
from src.model.resnet import resnet18


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self,shape):
        super(UnFlatten, self).__init__()
        self.shape=shape

    def forward(self, input):
        return input.view(input.size(0), *self.shape[1:])


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(out_channels, out_channels)


    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class VAE(nn.Module):
    def __init__(self, encoder=None, encoder_params=None, bilinear=True, image_channels=3, c_dim=64, h_dim=64*8*8, z_dim=64*8*8):
        super(VAE, self).__init__()
        self.encoder = resnet18()

        self.conv1 = nn.Conv2d(512, c_dim, kernel_size=1)  # 512-> 64
        self.conv2 = nn.Conv2d(512, c_dim, kernel_size=1)  # 512->64

        self.flatten = Flatten()  # 64x8x8

        self.unflatten = UnFlatten(shape=(-1, c_dim, 16, 16))  # 64*16*16 -> 64,16,16

        self.conv3 = nn.Conv2d(c_dim, 512, kernel_size=1)
        #

        self.decoder = nn.Sequential(
            Up(512, 256, bilinear),
            Up(256, 128, bilinear),
            Up(128, 64, bilinear),
            Up(64, 32, bilinear),
            nn.Conv2d(32, image_channels, kernel_size=1)
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # esp = torch.cuda.FloatTensor(*mu.size()).normal_()
        esp = torch.FloatTensor(*mu.size()).normal_()
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.conv1(h), self.conv2(h)
        mu = self.flatten(mu)
        logvar = self.flatten(logvar)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        z = self.unflatten(z)
        z = self.conv3(z)
        return self.decoder(z), mu, logvar


def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
