import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


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


class Decoder(nn.Module):
    def __init__(self, out_classes):
        super(Decoder, self).__init__()
        self.up_conv3_b = make_up_block(1024, 512)
        self.up_conv2_b = make_up_block(512, 256)
        self.up_conv1_b = make_up_block(256, 128)
        self.up_conv0_b = make_up_block(128, 64)

        self.conv_last_b = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
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


class ResNetBottleneck(nn.Module):
    def __init__(self, out_classes, resnet_type, is_autoencoder):
        super(ResNetBottleneck, self).__init__()

        # resnet layer4 channels
        self.in_channels = 1024
        self.is_autoencoder = is_autoencoder

        if resnet_type == 50:
            resnet = models.resnet50(pretrained=True)
            # remove fully connected layer, avg pool and layer5:
            self.encoder = nn.Sequential(*list(resnet.children())[:-3])

            print('pretrained resnet, 50')
        elif resnet_type == 101:
            resnet = models.resnet101(pretrained=True)
            # remove fully connected layer, avg pool and layer5:
            self.encoder = nn.Sequential(*list(resnet.children())[:-3])

            print('pretrained resnet, 101')
        elif resnet_type == 152:
            resnet = models.resnet152(pretrained=True)
            # remove fully connected layer, avg pool and layer5:
            self.encoder = nn.Sequential(*list(resnet.children())[:-3])

            print('pretrained resnet, 152')
        else:
            raise Exception('resnet_type must be in {50, 101, 152}!')

        # WORSE RESULTS
        self.layer4_up = self._make_up_block(Bottleneck, 512, 1, stride=2)
        self.layer3_up = self._make_up_block(Bottleneck, 256, 2, stride=2)
        self.layer2_up = self._make_up_block(Bottleneck, 128, 2, stride=2)
        self.layer1_up = self._make_up_block(Bottleneck, 64, 2, stride=2)

        self.out = nn.Conv2d(64, out_classes, kernel_size=1)

        # BETTER RESULTS
        self.decoder = Decoder(out_classes)

        if is_autoencoder:
            self.conv1 = nn.Conv2d(1024, 64, kernel_size=1)  # 1024->64
            self.conv2 = nn.Conv2d(1024, 64, kernel_size=1)  # 1024->64

            self.flatten = Flatten()  # 64x8x8

            self.unflatten = UnFlatten(shape=(-1, 64, 16, 16))

            self.conv3 = nn.Conv2d(64, 1024, kernel_size=1)

    def _make_up_block(self, block, init_channels, num_layer, stride=1):
        upsample = None
        # expansion = block.expansion
        if stride != 1 or self.in_channels != int(init_channels / 2):
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels,
                                   init_channels,
                                   kernel_size=1,
                                   stride=stride,
                                   output_padding=1),
                nn.BatchNorm2d(init_channels),
            )
        layers = []
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels))
        layers.append(block(self.in_channels, init_channels, stride, upsample))
        self.in_channels = init_channels
        return nn.Sequential(*layers)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.FloatTensor(*mu.size()).normal_()
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.conv1(h), self.conv2(h)
        mu = self.flatten(mu)
        logvar = self.flatten(logvar)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):

        h = self.encoder(x)

        # decoder worse results
        # out = self.layer4_up(out)
        # out = self.layer3_up(out)
        # out = self.layer2_up(out)
        # out = self.layer1_up(out)

        # out = self.out(out)

        # decoder
        if self.is_autoencoder:
            z, mu, logvar = self.bottleneck(h)
            z = self.unflatten(z)
            z = self.conv3(z)
            return self.decoder(z), mu, logvar

        else:
            return self.decoder(h)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, upsample=None):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if stride == 1:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1)
            self.conv3 = nn.Conv2d(out_channels, out_channels * 2,
                                   kernel_size=1)
            self.bn3 = nn.BatchNorm2d(out_channels * 2)

        else:
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels,
                                            kernel_size=3,
                                            stride=stride,
                                            padding=1,
                                            output_padding=1)
            self.conv3 = nn.Conv2d(out_channels, out_channels,
                                   kernel_size=1)
            self.bn3 = nn.BatchNorm2d(out_channels)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.upsample = upsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.ReLU()(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = nn.ReLU()(out)

        if self.upsample is not None:
            shortcut = self.upsample(x)

        out += shortcut

        return nn.ReLU()(out)


def ResNet50(out_classes=4, is_autoencoder=False):
    return ResNetBottleneck(out_classes, resnet_type=50, is_autoencoder=is_autoencoder)


if __name__ == '__main__':
    model = ResNet50()
    input_tensor = torch.rand(8, 3, 256, 256)

    output = model(input_tensor)
