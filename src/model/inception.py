import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_up_block(in_channels, out_channels):
    upblock = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

    return upblock


def make_first_block(in_channels, out_channels):
    upblock = nn.Sequential(
        nn.Upsample(scale_factor=3, mode='bilinear', align_corners=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

    return upblock


class Decoder(nn.Module):
    def __init__(self, out_classes):
        super(Decoder, self).__init__()
        self.up = make_first_block(2048, 1024)
        self.up_conv3_b = make_up_block(1024, 512)
        self.up_conv2_b = make_up_block(512, 256)
        self.up_conv1_b = make_up_block(256, 128)
        self.up_conv0_b = make_up_block(128, 64)

        self.conv_last_b = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        x = self.up(x)
        x = self.up_conv3_b(x)
        x = self.up_conv2_b(x)
        x = self.up_conv1_b(x)
        x = self.up_conv0_b(x)
        x = self.conv_last_b(x)
        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, shape):
        super(UnFlatten, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(input.size(0), *self.shape[1:])


class InceptionV3(nn.Module):
    def __init__(self, out_classes):
        super().__init__()

        self.inception = models.inception_v3(pretrained=True)
        self.inception.fc = Identity()
        self.inception.avgpool = Identity()
        self.inception.dropout = Identity()

        self.decoder = Decoder(out_classes)

    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.inception.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.inception.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.inception.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.inception.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.inception.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.inception.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.inception.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.inception.Mixed_7c(x)
        x = self.decoder(x)
        return x



if __name__ == '__main__':
    model = InceptionV3(8)
    input_tensor = torch.rand(8, 3, 256, 256)
    output = model(input_tensor)
    print(output)

