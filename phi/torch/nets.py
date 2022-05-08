from typing import Callable

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from . import TORCH
from ._torch_backend import register_module_call


def parameter_count(model: nn.Module) -> int:
    """
    Counts the number of parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        `int`
    """
    total = 0
    for parameter in model.parameters():
        total += numpy.prod(parameter.shape)
    return int(total)


def save_state(obj: nn.Module or optim.Optimizer, path: str):
    """
    Write the state of a module or optimizer to a file.

    See Also:
        `load_state()`

    Args:
        obj: `torch.nn.Module or torch.optim.Optimizer`
        path: File path as `str`.
    """
    if not path.endswith('.pth'):
        path += '.pth'
    torch.save(obj.state_dict(), path)


def load_state(obj: nn.Module or optim.Optimizer, path: str):
    """
    Read the state of a module or optimizer from a file.

    See Also:
        `save_state()`

    Args:
        obj: `torch.nn.Module or torch.optim.Optimizer`
        path: File path as `str`.
    """
    if not path.endswith('.pth'):
        path += '.pth'
    obj.load_state_dict(torch.load(path))


def update_weights(net: nn.Module, optimizer: optim.Optimizer, loss_function: Callable, *loss_args, **loss_kwargs):
    """
    Computes the gradients of `loss_function` w.r.t. the parameters of `net` and updates its weights using `optimizer`.

    This is the PyTorch version. Analogue functions exist for other learning frameworks.

    Args:
        net: Learning model.
        optimizer: Optimizer.
        loss_function: Loss function, called as `loss_function(*loss_args, **loss_kwargs)`.
        *loss_args: Arguments given to `loss_function`.
        **loss_kwargs: Keyword arguments given to `loss_function`.

    Returns:
        Output of `loss_function`.
    """
    optimizer.zero_grad()
    output = loss_function(*loss_args, **loss_kwargs)
    loss = output[0] if isinstance(output, tuple) else output
    loss.sum.backward()
    optimizer.step()
    return output


def adam(net: nn.Module, learning_rate: float = 1e-3, betas=(0.9, 0.999), epsilon=1e-07):
    """
    Creates an Adam optimizer for `net`, alias for [`torch.optim.Adam`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html).
    Analogue functions exist for other learning frameworks.
    """
    return optim.Adam(net.parameters(), learning_rate, betas, epsilon)


def dense_net(in_channels: int,
              out_channels: int,
              layers: tuple or list,
              batch_norm=False,
              activation: str or Callable = 'ReLU') -> nn.Module:
    layers = [in_channels, *layers, out_channels]
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    net = DenseNet(layers, activation, batch_norm)
    net = net.to(TORCH.get_default_device().ref)
    return net


class DenseNet(nn.Module):

    def __init__(self,
                 layers: list,
                 activation: type,
                 batch_norm: bool):
        super(DenseNet, self).__init__()
        self._layers = layers
        self._activation = activation
        self._batch_norm = batch_norm
        for i, (s1, s2) in enumerate(zip(layers[:-1], layers[1:])):
            self.add_module(f'linear{i}', nn.Linear(s1, s2, bias=True))
            if batch_norm:
                self.add_module(f'norm{i}', nn.BatchNorm1d(s2))

    def forward(self, x):
        register_module_call(self)
        x = TORCH.as_tensor(x)
        for i in range(len(self._layers) - 2):
            x = self._activation()(getattr(self, f'linear{i}')(x))
            if self._batch_norm:
                x = getattr(self, f'norm{i}')(x)
        x = getattr(self, f'linear{len(self._layers) - 2}')(x)
        return x


def u_net(in_channels: int,
          out_channels: int,
          levels: int = 4,
          filters: int or tuple or list = 16,
          batch_norm: bool = True,
          activation: str or type = 'ReLU',
          in_spatial: tuple or int = 2) -> nn.Module:
    if isinstance(filters, (tuple, list)):
        assert len(filters) == levels, f"List of filters has length {len(filters)} but u-net has {levels} levels."
    else:
        filters = (filters,) * levels
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    if isinstance(in_spatial, int):
        d = in_spatial
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)
    net = UNet(d, in_channels, out_channels, filters, batch_norm, activation)
    net = net.to(TORCH.get_default_device().ref)
    # net = torch.jit.trace_module(net, {'forward': torch.zeros((1, in_channels) + (32,) * d, device=TORCH.get_default_device().ref)})
    return net


class UNet(nn.Module):

    def __init__(self, d: int, in_channels: int, out_channels: int, filters: tuple, batch_norm: bool, activation: type):
        super(UNet, self).__init__()
        self._levels = len(filters)
        self._spatial_rank = d
        self.add_module('inc', DoubleConv(d, in_channels, filters[0], filters[0], batch_norm, activation))
        for i in range(1, self._levels):
            self.add_module(f'down{i}', Down(d, filters[i - 1], filters[i], batch_norm, activation))
            self.add_module(f'up{i}', Up(d, filters[i] + filters[i - 1], filters[i - 1], batch_norm, activation))
        self.add_module('outc', CONV[d](filters[0], out_channels, kernel_size=1))

    def forward(self, x):
        register_module_call(self)
        x = TORCH.as_tensor(x)
        x = self.inc(x)
        xs = [x]
        for i in range(1, self._levels):
            x = getattr(self, f'down{i}')(x)
            xs.insert(0, x)
        for i in range(1, self._levels):
            x = getattr(self, f'up{i}')(x, xs[i])
        x = self.outc(x)
        return x


# class ConvDenseNet(nn.Module):
#
#     def __init__(self,
#                  in_channels: int,
#                  in_spatial: tuple,
#                  filters: tuple,
#                  dense_layers: list,
#                  d=2):
#         super(ConvDenseNet, self).__init__()
#         # Conv
#         self._levels = len(filters)
#         self.inc = DoubleConv(in_channels, filters[0], d=d)
#         for i in range(1, self._levels):
#             self.add_module(f'conv{i}', Down(filters[i - 1], filters[i], d=d))
#         # Dense
#         in_neurons = int(numpy.prod(in_spatial) * filters[-1] / 2 ** (d * (len(filters) - 1)))
#         self._dense_layers = (in_neurons, *dense_layers)
#         for i, (s1, s2) in enumerate(zip(self._dense_layers[:-1], self._dense_layers[1:])):
#             self.add_module(f'linear{i}', nn.Linear(s1, s2, bias=True))
#
#     def forward(self, x):
#         x = self.inc(x)
#         for i in range(1, self._levels):
#             x = getattr(self, f'conv{i}')(x)
#         x = torch.reshape(x, (x.shape[0], -1))
#         for i in range(len(self._dense_layers) - 2):
#             x = F.relu(getattr(self, f'linear{i}')(x))
#         x = getattr(self, f'linear{len(self._dense_layers) - 2}')(x)
#         return x


CONV = [None, nn.Conv1d, nn.Conv2d, nn.Conv3d]
NORM = [None, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
ACTIVATIONS = {'ReLU': nn.ReLU, 'Sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, d: int, in_channels: int, out_channels: int, mid_channels: int, batch_norm: bool, activation: type):
        super().__init__()
        self.add_module('double_conv', nn.Sequential(
            CONV[d](in_channels, mid_channels, kernel_size=3, padding=1),
            NORM[d](mid_channels) if batch_norm else nn.Identity(),
            activation(),
            CONV[d](mid_channels, out_channels, kernel_size=3, padding=1),
            NORM[d](out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True)
        ))

    def forward(self, x):
        return self.double_conv(x)


MAX_POOL = [None, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, d: int, in_channels: int, out_channels: int, batch_norm: bool, activation: type):
        super().__init__()
        self.add_module('maxpool', MAX_POOL[d](2))
        self.add_module('conv', DoubleConv(d, in_channels, out_channels, out_channels, batch_norm, activation))

    def forward(self, x):
        x = self.maxpool(x)
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    _MODES = [None, 'linear', 'bilinear', 'trilinear']

    def __init__(self, d: int, in_channels: int, out_channels: int, batch_norm: bool, activation: type, linear=True):
        super().__init__()
        if linear:
            # if bilinear, use the normal convolutions to reduce the number of channels
            up = nn.Upsample(scale_factor=2, mode=Up._MODES[d])
            conv = DoubleConv(d, in_channels, out_channels, in_channels // 2, batch_norm, activation)
        else:
            up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            conv = DoubleConv(d, in_channels, out_channels, out_channels, batch_norm, activation)
        self.add_module('up', up)
        self.add_module('conv', conv)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        # diff = [x2.size()[i] - x1.size()[i] for i in range(2, len(x1.shape))]
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
