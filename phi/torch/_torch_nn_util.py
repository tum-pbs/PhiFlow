import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import TORCH


def parameter_count(model: nn.Module):
    total = 0
    for parameter in model.parameters():
        total += numpy.prod(parameter.shape)
    return int(total)


def dense_net(in_channels: int,
              out_channels: int,
              layers: tuple or list,
              batch_norm=False) -> nn.Module:
    if batch_norm:
        raise NotImplementedError("only batch_norm=False currently supported")
    layers = [in_channels, *layers, out_channels]
    return DenseNet(layers)


class DenseNet(nn.Module):

    def __init__(self,
                 layers: list):
        super(DenseNet, self).__init__()
        self._layers = layers
        for i, (s1, s2) in enumerate(zip(layers[:-1], layers[1:])):
            self.add_module(f'linear{i}', nn.Linear(s1, s2, bias=True))

    def forward(self, x):
        for i in range(len(self._layers) - 2):
            x = F.relu(getattr(self, f'linear{i}')(x))
        x = getattr(self, f'linear{len(self._layers) - 2}')(x)
        return x


def u_net(in_channels: int,
          out_channels: int,
          levels: int = 4,
          filters: int or tuple or list = 16,
          batch_norm=True,
          in_spatial=None) -> nn.Module:
    if not batch_norm:
        raise NotImplementedError("only batch_norm=True currently supported")
    if isinstance(filters, (tuple, list)):
        assert len(filters) == levels, f"List of filters has length {len(filters)} but u-net has {levels} levels."
    else:
        filters = (filters,) * levels
    net = UNet(in_channels, out_channels, filters)
    net = net.to(TORCH.get_default_device().ref)
    return net


class UNet(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 filter_counts: tuple,
                 batch_norm=True):
        super(UNet, self).__init__()
        assert batch_norm, "Not yet implemented"  # TODO
        self._levels = len(filter_counts)
        self.inc = DoubleConv(in_channels, filter_counts[0])
        for i in range(1, self._levels):
            self.add_module(f'down{i}', Down(filter_counts[i - 1], filter_counts[i]))
            self.add_module(f'up{i}', Up(filter_counts[i] + filter_counts[i-1], filter_counts[i - 1]))
        self.outc = OutConv(filter_counts[0], out_channels)

    def forward(self, x):
        x = self.inc(x)
        xs = [x]
        for i in range(1, self._levels):
            xs.insert(0, getattr(self, f'down{i}')(x))
        x = xs[0]
        for i in range(1, self._levels):
            x = getattr(self, f'up{i}')(x, xs[i])
        x = self.outc(x)
        return x


class ConvDenseNet(nn.Module):

    def __init__(self,
                 in_channels: int,
                 in_spatial: tuple,
                 filter_counts: tuple,
                 dense_layers: list,
                 d=2):
        super(ConvDenseNet, self).__init__()
        # Conv
        self._levels = len(filter_counts)
        self.inc = DoubleConv(in_channels, filter_counts[0], d=d)
        for i in range(1, self._levels):
            self.add_module(f'conv{i}', Down(filter_counts[i - 1], filter_counts[i], d=d))
        # Dense
        in_neurons = int(numpy.prod(in_spatial) * filter_counts[-1] / 2 ** (d * (len(filter_counts) - 1)))
        self._dense_layers = (in_neurons, *dense_layers)
        for i, (s1, s2) in enumerate(zip(self._dense_layers[:-1], self._dense_layers[1:])):
            self.add_module(f'linear{i}', nn.Linear(s1, s2, bias=True))

    def forward(self, x):
        x = self.inc(x)
        for i in range(1, self._levels):
            x = getattr(self, f'conv{i}')(x)
        x = torch.reshape(x, (x.shape[0], -1))
        for i in range(len(self._dense_layers) - 2):
            x = F.relu(getattr(self, f'linear{i}')(x))
        x = getattr(self, f'linear{len(self._dense_layers) - 2}')(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, d=2):
        super().__init__()
        if d == 2:
            conv = nn.Conv2d
            norm = nn.BatchNorm2d
        elif d == 1:
            conv = nn.Conv1d
            norm = nn.BatchNorm1d
        elif d == 3:
            conv = nn.Conv3d
            norm = nn.BatchNorm3d
        else:
            raise ValueError(f"{d} dimensions are not supported, must be [1,3].")
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            conv(in_channels, mid_channels, kernel_size=3, padding=1),
            norm(mid_channels),
            nn.ReLU(inplace=True),
            conv(mid_channels, out_channels, kernel_size=3, padding=1),
            norm(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, d=2):
        super().__init__()
        if d == 1:
            maxpool = nn.MaxPool1d
        elif d == 2:
            maxpool = nn.MaxPool2d
        elif d == 3:
            maxpool = nn.MaxPool3d
        else:
            raise ValueError(f"{d} dimensions are not supported, must be [1,3].")

        self.maxpool_conv = nn.Sequential(
            maxpool(2),
            DoubleConv(in_channels, out_channels, d=d)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
