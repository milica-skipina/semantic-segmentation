import torch
from torch import nn
from torch.nn import functional as F


class EncoderConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(EncoderConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv_block(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = EncoderConvBlock(3, 64, 7, 2, 3)
        self.conv2 = EncoderConvBlock(64, 128, 5, 2, 2)
        self.conv3 = EncoderConvBlock(128, 256, 3, 2, 1)
        self.conv4 = EncoderConvBlock(256, 512, 3, 2, 1)

    def forward(self, x):
        x = self.conv1(x)  # 128
        x = self.conv2(x)  # 64
        x = self.conv3(x)  # 32
        x = self.conv4(x)  # 16
        return x


class DecoderConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, output_padding=0):
        super(DecoderConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            #nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, x):
        return self.conv_block(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = DecoderConvBlock(512, 256)  # 3, 2, 1, 1)
        self.conv2 = DecoderConvBlock(256, 128)  # 3, 2, 1, 1)
        self.conv3 = DecoderConvBlock(128, 64)  # 5, 2, 2, 1)
        self.conv4 = DecoderConvBlock(64, 3)  # 7, 2, 3, 1)

    def forward(self, x):
        x = self.conv1(x)  # 32
        x = self.conv2(x)  # 64
        x = self.conv3(x)  # 128
        x = self.conv4(x)  # 256
        return x


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2),  # N, 27, 27
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 5, stride=3, output_padding=1),  # N, 84, 84
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 3, 7, stride=3),  # N, 256, 256
            nn.Sigmoid()
        )

        self.decoder = Decoder()

    def forward(self, x):
        out = self.encoder(x)
        return self.decoder(out)


if __name__ == '__main__':
    input_torch = torch.rand(8, 3, 256, 256)
    net = ConvAutoencoder()
    out = net(input_torch)
    print('haha')