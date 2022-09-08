from typing import Callable, Dict, List, Tuple

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from .. import math
from . import TORCH
from ._torch_backend import register_module_call
from ..math import channel


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


def get_parameters(net: nn.Module, wrap=True) -> dict:
    if not wrap:
        return {name: param for name, param in net.named_parameters()}
    result = {}
    for name, param in net.named_parameters():
        if name.endswith('.weight'):
            phi_tensor = math.wrap(param, channel('input,output'))
        elif name.endswith('.bias'):
            phi_tensor = math.wrap(param, channel('output'))
        else:
            raise NotImplementedError
        result[name] = phi_tensor
    return result


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


def sgd(net: nn.Module, learning_rate: float = 1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False):
    """
    Creates an SGD optimizer for 'net', alias for ['torch.optim.SGD'](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
    Analogue functions exist for other learning frameworks.
    """
    return optim.SGD(net.parameters(), learning_rate, momentum, dampening, weight_decay, nesterov)


def adagrad(net: nn.Module, learning_rate: float = 1e-3, lr_decay=0, weight_decay=0, initial_accumulator_value=0,
            eps=1e-10):
    """
    Creates an Adagrad optimizer for 'net', alias for ['torch.optim.Adagrad'](https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html)
    Analogue functions exist for other learning frameworks.
    """
    return optim.Adagrad(net.parameters(), learning_rate, lr_decay, weight_decay, initial_accumulator_value, eps)


def rmsprop(net: nn.Module, learning_rate: float = 1e-3, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0,
            centered=False):
    """
    Creates an RMSProp optimizer for 'net', alias for ['torch.optim.RMSprop'](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html)
    Analogue functions exist for other learning frameworks.
    """
    return optim.RMSprop(net.parameters(), learning_rate, alpha, eps, weight_decay, momentum, centered)


CONV = [None, nn.Conv1d, nn.Conv2d, nn.Conv3d]
NORM = [None, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
ACTIVATIONS = {'ReLU': nn.ReLU, 'Sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'SiLU': nn.SiLU}


def dense_net(in_channels: int,
              out_channels: int,
              layers: Tuple[int, ...] or List[int],
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
          in_spatial: tuple or int = 2,
          use_res_blocks: bool = False) -> nn.Module:
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
    net = UNet(d, in_channels, out_channels, filters, batch_norm, activation, use_res_blocks)
    net = net.to(TORCH.get_default_device().ref)
    # net = torch.jit.trace_module(net, {'forward': torch.zeros((1, in_channels) + (32,) * d, device=TORCH.get_default_device().ref)})
    return net


class UNet(nn.Module):

    def __init__(self, d: int, in_channels: int, out_channels: int, filters: tuple, batch_norm: bool, activation: type,
                 use_res_blocks: bool):
        super(UNet, self).__init__()
        self._levels = len(filters)
        self._spatial_rank = d
        if use_res_blocks:
            self.add_module('inc', ResNet_Block(d, in_channels, filters[0], batch_norm, activation))
        else:
            self.add_module('inc', DoubleConv(d, in_channels, filters[0], filters[0], batch_norm, activation))
        for i in range(1, self._levels):
            self.add_module(f'down{i}', Down(d, filters[i - 1], filters[i], batch_norm, activation, use_res_blocks))
            self.add_module(f'up{i}', Up(d, filters[i] + filters[i - 1], filters[i - 1], batch_norm, activation,
                                         use_res_blocks=use_res_blocks))
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


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, d: int, in_channels: int, out_channels: int, mid_channels: int, batch_norm: bool,
                 activation: type):
        super().__init__()
        self.add_module('double_conv', nn.Sequential(
            CONV[d](in_channels, mid_channels, kernel_size=3, padding=1, padding_mode='circular'),
            NORM[d](mid_channels) if batch_norm else nn.Identity(),
            activation(),
            CONV[d](mid_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular'),
            NORM[d](out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True)
        ))

    def forward(self, x):
        return self.double_conv(x)


MAX_POOL = [None, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]


class Down(nn.Module):
    """Downscaling with maxpool then double conv or resnet_block"""

    def __init__(self, d: int, in_channels: int, out_channels: int, batch_norm: bool, activation: str or type,
                 use_res_blocks: bool):
        super().__init__()
        self.add_module('maxpool', MAX_POOL[d](2))
        if use_res_blocks:
            self.add_module('conv', ResNet_Block(d, in_channels, out_channels, batch_norm, activation))
        else:
            self.add_module('conv', DoubleConv(d, in_channels, out_channels, out_channels, batch_norm, activation))

    def forward(self, x):
        x = self.maxpool(x)
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    _MODES = [None, 'linear', 'bilinear', 'trilinear']

    def __init__(self, d: int, in_channels: int, out_channels: int, batch_norm: bool, activation: type, linear=True,
                 use_res_blocks: bool = False):
        super().__init__()
        if linear:
            # if bilinear, use the normal convolutions to reduce the number of channels
            up = nn.Upsample(scale_factor=2, mode=Up._MODES[d])
            if use_res_blocks:
                conv = ResNet_Block(d, in_channels, out_channels, batch_norm, activation)
            else:
                conv = DoubleConv(d, in_channels, out_channels, in_channels // 2, batch_norm, activation)
        else:
            up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            if use_res_blocks:
                conv = ResNet_Block(d, in_channels, out_channels, batch_norm, activation)
            else:
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


class ConvNet(nn.Module):

    def __init__(self, in_spatial, in_channels, out_channels, layers, batch_norm, activation):
        super(ConvNet, self).__init__()
        self.layers = layers
        activation = ACTIVATIONS[activation]

        self.add_module(f'Conv_in', nn.Sequential(
            CONV[in_spatial](in_channels, layers[0], kernel_size=3, padding=1, padding_mode='circular'),
            NORM[in_spatial](layers[0]) if batch_norm else nn.Identity(),
            activation()))
        for i in range(1, len(layers)):
            self.add_module(f'Conv{i}', nn.Sequential(
                CONV[in_spatial](layers[i - 1], layers[i], kernel_size=3, padding=1, padding_mode='circular'),
                NORM[in_spatial](layers[i]) if batch_norm else nn.Identity(),
                activation()))
        self.add_module(f'Conv_out', CONV[in_spatial](layers[len(layers) - 1], out_channels, kernel_size=3, padding=1, padding_mode='circular'))

    def forward(self, x):
        x = getattr(self, f'Conv_in')(x)
        for i in range(1, len(self.layers)):
            x = getattr(self, f'Conv{i}')(x)
        x = getattr(self, f'Conv_out')(x)
        return x


def conv_net(in_channels: int,
             out_channels: int,
             layers: Tuple[int, ...] or List[int],
             batch_norm: bool = False,
             activation: str or type = 'ReLU',
             in_spatial: int or tuple = 2) -> nn.Module:
    if isinstance(in_spatial, int):
        d = in_spatial
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)
    net = ConvNet(d, in_channels, out_channels, layers, batch_norm, activation)
    net = net.to(TORCH.get_default_device().ref)
    return net


class ResNet_Block(nn.Module):

    def __init__(self, in_spatial, in_channels, out_channels, batch_norm, activation):
        # Since in_channels and out_channels might be different
        # we need a sampling layer for up/down sampling input
        # in order to add it as a skip connection
        super(ResNet_Block, self).__init__()
        if in_channels != out_channels:
            self.sample_input = CONV[in_spatial](in_channels, out_channels, kernel_size=1, padding=0)
            self.bn_sample = NORM[in_spatial](out_channels) if batch_norm else nn.Identity()
        else:
            self.sample_input = nn.Identity()
            self.bn_sample = nn.Identity()

        self.activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation

        self.bn1 = NORM[in_spatial](out_channels) if batch_norm else nn.Identity()
        self.conv1 = CONV[in_spatial](in_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular')

        self.bn2 = NORM[in_spatial](out_channels) if batch_norm else nn.Identity()
        self.conv2 = CONV[in_spatial](out_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular')

    def forward(self, x):
        x = TORCH.as_tensor(x)
        out = self.activation()(self.bn1(self.conv1(x)))

        out = self.activation()(self.bn2(self.conv2(out)))

        out = (out + self.bn_sample(self.sample_input(x)))

        return out

class Dense_ResNet_Block(nn.Module):

    def __init__(self, in_channels, mid_channels, batch_norm, activation):
        super(Dense_ResNet_Block, self).__init__()

        self.bn1 = NORM[1](mid_channels) if batch_norm else nn.Identity()
        self.linear1 = nn.Linear(in_channels, mid_channels)

        self.bn2 = NORM[1](in_channels) if batch_norm else nn.Identity()
        self.linear2 = nn.Linear(mid_channels, in_channels)

    def forward(self, x):
        x = TORCH.as_tensor(x)
        out = self.activation()(self.bn1(self.linear1(x)))

        out = self.activation()(self.bn2(self.linear2(out)))

        out = out + x

        return out

def get_mask(inputs, reverse_mask, data_format = 'NHWC'):
    shape = inputs.shape
    if len(shape) == 2:
        N = shape[-1]
        range_n = torch.arange(0, N)
        even_ind = range_n % 2
        checker = torch.reshape(even_ind, (-1, N))
    elif len(shape) == 4:
        H = shape[2] if data_format == 'NCHW' else shape[1]
        W = shape[3] if data_format == 'NCHW' else shape[2]

        range_h = torch.arange(0, H)
        range_w = torch.arange(0, W)

        even_ind_h = range_h % 2
        even_ind_w = range_w % 2

        ind_h = even_ind_h.unsqueeze(-1).repeat(1, W)
        ind_w = even_ind_w.unsqueeze( 0).repeat(H, 1)

        checker = torch.logical_xor(ind_h, ind_w)

        checker = checker.reshape(1, 1, H, W) if data_format == 'NCHW' else checker.reshape(1, H, W, 1)
        checker = checker.long()

    else:
        raise ValueError('Invalid tensor shape. Dimension of the tensor shape must be '
                         '2 (NxD) or 4 (NxCxHxW or NxHxWxC), got {}.'.format(inputs.get_shape().as_list()))

    if reverse_mask:
        checker = 1 - checker

    return checker.to(TORCH.get_default_device().ref)

class CouplingLayer(nn.Module):

    def __init__(self, in_channels, mid_channels, activation, batch_norm, in_spatial, reverse_mask):
        super(CouplingLayer, self).__init__()

        self.activation = activation
        self.batch_norm = batch_norm
        self.reverse_mask = reverse_mask


        if in_spatial == 0: #for in_spatial = 0, use dense layers
            self.s1 = nn.Sequential(Dense_ResNet_Block(in_channels, mid_channels, batch_norm, activation),
                                    torch.nn.Tanh())
            self.t1 = Dense_ResNet_Block(in_channels, mid_channels, batch_norm, activation)

            self.s2 = nn.Sequential(Dense_ResNet_Block(in_channels, mid_channels, batch_norm, activation),
                                    torch.nn.Tanh())
            self.t2 = Dense_ResNet_Block(in_channels, mid_channels, batch_norm, activation)
        else:
            self.s1 = nn.Sequential(ResNet_Block(in_spatial, in_channels, in_channels, batch_norm, activation),
                                    torch.nn.Tanh())
            self.t1 = ResNet_Block(in_spatial, in_channels, in_channels, batch_norm, activation)

            self.s2 = nn.Sequential(ResNet_Block(in_spatial, in_channels, in_channels, batch_norm, activation),
                                    torch.nn.Tanh())
            self.t2 = ResNet_Block(in_spatial, in_channels, in_channels, batch_norm, activation)


    def forward(self, x, invert=False):
        x = TORCH.as_tensor(x)
        mask = get_mask(x, self.reverse_mask, 'NCHW')

        if invert:
            v1 = x * mask
            v2 = x * (1-mask)

            u2 = (1-mask) * (v2 - self.t1(v1)) * torch.exp(-self.s1(v1))
            u1 = mask * (v1 - self.t2(u2)) * torch.exp(-self.s2(u2))

            return u1 + u2
        else:
            u1 = x * mask
            u2 = x * (1-mask)

            v1 = mask * (u1 * torch.exp( self.s2(u2)) + self.t2(u2))
            v2 = (1-mask) * (u2 * torch.exp( self.s1(v1)) + self.t1(v1))

            return v1 + v2

class Inn(nn.Module):
    def __init__(self, in_channels, mid_channels, num_blocks, activation, batch_norm, in_spatial, reverse_mask):
        super(Inn, self).__init__()
        self.num_blocks = num_blocks

        for i in range(num_blocks):
            self.add_module(f'coupling_block{i+1}', CouplingLayer(in_channels, mid_channels, activation, batch_norm, in_spatial, reverse_mask))

    def forward(self, x, backward=False):
        if backward:
            for i in range(self.num_blocks, 0, -1):
                x = getattr(self, f'coupling_block{i}')(x, backward)
        else:
            for i in range(1, self.num_blocks+1):
                x = getattr(self, f'coupling_block{i}')(x, backward)
        return x


def inn(in_channels: int,
        mid_channels: int,
        num_blocks: int,
        batch_norm: bool = False,
        reverse_mask: bool = False,
        activation: str or type='ReLU',
        in_spatial: tuple or int=2,
        backward: bool = False):
    if isinstance(in_spatial, tuple):
        in_spatial = len(in_spatial)

    return Inn(in_channels, mid_channels, num_blocks, activation, batch_norm, in_spatial, reverse_mask).to(TORCH.get_default_device().ref)


def coupling_layer(in_channels: int,
                   mid_channels: int,
                   activation: str or type='ReLU',
                   batch_norm=False,
                   reverse_mask=False,
                   in_spatial: tuple or int=2):
    if isinstance(in_spatial, tuple):
        in_spatial = len(in_spatial)

    net = CouplingLayer(in_channels, mid_channels, activation, batch_norm, in_spatial, reverse_mask)
    net = net.to(TORCH.get_default_device().ref)
    return net

class ResNet(nn.Module):

    def __init__(self, in_spatial, in_channels, out_channels, layers, batch_norm, activation):
        super(ResNet, self).__init__()
        self.layers = layers

        if len(self.layers) >0:
            self.add_module('Res_in', ResNet_Block(in_spatial, in_channels, layers[0], batch_norm, activation))

            for i in range(1, len(layers)):
                self.add_module(f'Res{i}', ResNet_Block(in_spatial, layers[i-1], layers[i], batch_norm, activation))

            self.add_module('Res_out', ResNet_Block(in_spatial, layers[len(layers)-1], out_channels, batch_norm, activation))
        else:
            self.add_module('Res', ResNet_Block(in_spatial, in_channels, out_channels, batch_norm, activation))

    def forward(self, x):
        x = TORCH.as_tensor(x)
        if len(self.layers) > 0:
            x = getattr(self, 'Res_in')(x)
            for i in range(1, len(self.layers)):
                x = getattr(self, f'Res{i}')(x)
            x = getattr(self, 'Res_out')(x)
        else:
            print('Single Res block')
            x = getattr(self, 'Res')(x)
        return x


def res_net(in_channels: int,
            out_channels: int,
            layers: Tuple[int, ...] or List[int],
            batch_norm: bool = False,
            activation: str or type = 'ReLU',
            in_spatial: int or tuple = 2) -> nn.Module:
    if (isinstance(in_spatial, int)):
        d = in_spatial
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)
    net = ResNet(d, in_channels, out_channels, layers, batch_norm, activation)
    net = net.to(TORCH.get_default_device().ref)
    return net


def conv_classifier(input_shape: list, num_classes: int, batch_norm: bool, in_spatial: int or tuple):
    if isinstance(in_spatial, int):
        d = in_spatial
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)
    net = ConvClassifier(d, input_shape, num_classes, batch_norm)
    net = net.to(TORCH.get_default_device().ref)
    return net


class ConvClassifier(nn.Module):

    def __init__(self, d: int, input_shape: list, num_classes: int, batch_norm: bool):
        super(ConvClassifier, self).__init__()

        self.spatial_shape_list = list(input_shape[1:])
        self.add_module('maxpool', MAX_POOL[d](2))

        self.add_module('conv1', DoubleConv(d, input_shape[0], 64, 64, batch_norm, ACTIVATIONS['ReLU']))

        self.add_module('conv2', DoubleConv(d, 64, 128, 128, batch_norm, ACTIVATIONS['ReLU']))

        self.add_module('conv3', nn.Sequential(DoubleConv(d, 128, 256, 256, batch_norm, ACTIVATIONS['ReLU']),
                                               CONV[d](256, 256, 3, padding=1, padding_mode='circular'),
                                               NORM[d](256) if batch_norm else nn.Identity(),
                                               nn.ReLU()))

        self.add_module('conv4', nn.Sequential(DoubleConv(d, 256, 512, 512, batch_norm, ACTIVATIONS['ReLU']),
                                               CONV[d](512, 512, 3, padding=1, padding_mode='circular'),
                                               NORM[d](512) if batch_norm else nn.Identity(),
                                               nn.ReLU()))

        self.add_module('conv5', nn.Sequential(DoubleConv(d, 512, 512, 512, batch_norm, ACTIVATIONS['ReLU']),
                                               CONV[d](512, 512, 3, padding=1, padding_mode='circular'),
                                               NORM[d](512) if batch_norm else nn.Identity(),
                                               nn.ReLU()))

        for i in range(5):
            for j in range(len(self.spatial_shape_list)):
                self.spatial_shape_list[j] = math.floor((self.spatial_shape_list[j] - 2) / 2) + 1

        flattened_input_dim = 1
        for i in range(len(self.spatial_shape_list)):
            flattened_input_dim *= self.spatial_shape_list[i]
        flattened_input_dim *= 512

        self.linear = dense_net(flattened_input_dim, num_classes, [4096, 4096, 100], batch_norm, 'ReLU')
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax()

    def forward(self, x):

        for i in range(5):
            x = getattr(self, f'conv{i + 1}')(x)
            x = self.maxpool(x)
        x = self.flatten(x)
        x = self.softmax(self.linear(x))
        return x
