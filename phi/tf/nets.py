"""
Jax implementation of the unified machine learning API.
Equivalent functions also exist for the other frameworks.

For API documentation, see https://tum-pbs.github.io/PhiFlow/Network_API .
"""
from typing import Callable, Tuple, List
import pickle
from typing import Callable
from typing import Tuple, List

import numpy
import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow import keras
from tensorflow.keras import layers as kl
from tensorflow import Tensor

from .. import math


def parameter_count(model: keras.Model):
    """
    Counts the number of parameters in a model.

    Args:
        model: Keras model

    Returns:
        `int`
    """
    total = 0
    for parameter in model.trainable_weights:
        total += numpy.prod(parameter.shape)
    return int(total)


def get_parameters(model: keras.Model, wrap=True) -> dict:
    result = {}
    for var in model.trainable_weights:
        name: str = var.name
        layer = name[:name.index('/')].replace('_', '').replace('dense', 'linear')
        try:
            int(layer[-1:])
        except ValueError:
            layer += '0'
        prop = name[name.index('/') + 1:].replace('kernel', 'weight')
        if prop.endswith(':0'):
            prop = prop[:-2]
        name = f"{layer}.{prop}"
        var = var.numpy()
        if not wrap:
            result[name] = var
        else:
            if name.endswith('.weight'):
                if var.ndim == 2:
                    phi_tensor = math.wrap(var, math.channel('input,output'))
                elif var.ndim == 3:
                    phi_tensor = math.wrap(var, math.channel('x,input,output'))
                elif var.ndim == 4:
                    phi_tensor = math.wrap(var, math.channel('x,y,input,output'))
                elif var.ndim == 5:
                    phi_tensor = math.wrap(var, math.channel('x,y,z,input,output'))
            elif name.endswith('.bias'):
                phi_tensor = math.wrap(var, math.channel('output'))
            elif var.ndim == 1:
                phi_tensor = math.wrap(var, math.channel('output'))
            else:
                raise NotImplementedError(name, var)
            result[name] = phi_tensor
    return result


def save_state(obj: keras.models.Model or keras.optimizers.Optimizer, path: str):
    """
    Write the state of a module or optimizer to a file.

    See Also:
        `load_state()`

    Args:
        obj: `keras.models.Model or keras.optimizers.Optimizer`
        path: File path as `str`.
    """
    if isinstance(obj, keras.models.Model):
        if not path.endswith('.h5'):
            path += '.h5'
        obj.save_weights(path)
    elif isinstance(obj, keras.optimizers.Optimizer):
        if not path.endswith('.pkl'):
            path += '.pkl'
        weights = obj.get_weights()
        with open(path, 'wb') as f:
            pickle.dump(weights, f)
    else:
        raise ValueError("obj must be a Keras model or optimizer")


def load_state(obj: keras.models.Model or keras.optimizers.Optimizer, path: str):
    """
    Read the state of a module or optimizer from a file.

    See Also:
        `save_state()`

    Args:
        obj: `keras.models.Model or keras.optimizers.Optimizer`
        path: File path as `str`.
    """
    if isinstance(obj, keras.models.Model):
        if not path.endswith('.h5'):
            path += '.h5'
        obj.load_weights(path)
    elif isinstance(obj, keras.optimizers.Optimizer):
        if not path.endswith('.pkl'):
            path += '.pkl'
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        obj.set_weights(weights)
    else:
        raise ValueError("obj must be a Keras model or optimizer")


def update_weights(net: keras.Model, optimizer: keras.optimizers.Optimizer, loss_function: Callable, *loss_args, **loss_kwargs):
    """
    Computes the gradients of `loss_function` w.r.t. the parameters of `net` and updates its weights using `optimizer`.

    This is the TensorFlow/Keras version. Analogue functions exist for other learning frameworks.

    Args:
        net: Learning model.
        optimizer: Optimizer.
        loss_function: Loss function, called as `loss_function(*loss_args, **loss_kwargs)`.
        *loss_args: Arguments given to `loss_function`.
        **loss_kwargs: Keyword arguments given to `loss_function`.

    Returns:
        Output of `loss_function`.
    """
    with tf.GradientTape() as tape:
        output = loss_function(*loss_args, **loss_kwargs)
        loss = output[0] if isinstance(output, tuple) else output
        gradients = tape.gradient(loss.sum, net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, net.trainable_variables))
    return output


def adam(net: keras.Model, learning_rate: float = 1e-3, betas=(0.9, 0.999), epsilon=1e-07):
    """
    Creates an Adam optimizer for `net`, alias for [`keras.optimizers.Adam`](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam).
    Analogous functions exist for other learning frameworks.
    """
    return keras.optimizers.Adam(learning_rate, betas[0], betas[1], epsilon)


def sgd(net: keras.Model, learning_rate: float = 1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False):
    """
    Creates an SGD optimizer for 'net', alias for ['keras.optimizers.SGD'](https://keras.io/api/optimizers/sgd/)
    Analogous functions exist for other learning frameworks.
    """
    return keras.optimizers.SGD(learning_rate, momentum, nesterov)


def adagrad(net: keras.Model, learning_rate: float = 1e-3, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10):
    """
    Creates an Adagrad optimizer for 'net', alias for ['keras.optimizers.Adagrad'](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adagrad)
    Analogous functions exist for other learning frameworks.
    """
    return keras.optimizers.Adagrad(learning_rate, initial_accumulator_value, eps)


def rmsprop(net: keras.Model, learning_rate: float = 1e-3, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False):
    """
    Creates an RMSProp optimizer for 'net', alias for ['keras.optimizers.RMSprop'](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop)
    Analogous functions exist for other learning frameworks.
    """
    return keras.optimizers.RMSprop(learning_rate, alpha, momentum, eps, centered)


def dense_net(in_channels: int,
              out_channels: int,
              layers: Tuple[int, ...] or List[int],
              batch_norm=False,
              activation='ReLU',
              softmax=False) -> keras.Model:
    """
    Fully-connected neural networks are available in ΦFlow via dense_net().
    Arguments:
        in_channels : size of input layer, int
        out_channels = size of output layer, int
        layers : tuple of linear layers between input and output neurons, list or tuple
        activation : activation function used within the layers, string
        batch_norm : use of batch norm after each linear layer, bool

    Returns:
        Dense net model as specified by input arguments
    """
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    keras_layers = []
    for neuron_count in layers:
        keras_layers.append(kl.Dense(neuron_count, activation=activation))
        if batch_norm:
            keras_layers.append(kl.BatchNormalization())
    return keras.models.Sequential([kl.InputLayer(input_shape=(in_channels,)),
                                    *keras_layers,
                                    kl.Dense(out_channels, activation='linear'),
                                    *([kl.Softmax()] if softmax else [])])


def u_net(in_channels: int,
          out_channels: int,
          levels: int = 4,
          filters: int or tuple or list = 16,
          batch_norm: bool = True,
          activation: str or Callable = 'ReLU',
          in_spatial: tuple or int = 2,
          periodic=False,
          use_res_blocks: bool = False, **kwargs) -> keras.Model:
    """
    ΦFlow provides a built-in U-net architecture, classically popular for Semantic Segmentation in Computer Vision, composed of downsampling and upsampling layers.

    Arguments:

        in_channels: input channels of the feature map, dtype : int
        out_channels : output channels of the feature map, dtype : int
        levels : number of levels of down-sampling and upsampling, dtype : int
        filters : filter sizes at each down/up sampling convolutional layer, if the input is integer all conv layers have the same filter size,
        dtype : int or tuple
        activation : activation function used within the layers, dtype : string
        batch_norm : use of batchnorm after each conv layer, dtype : bool
        in_spatial : spatial dimensions of the input feature map, dtype : int
        use_res_blocks : use convolutional blocks with skip connections instead of regular convolutional blocks, dtype : bool

    Returns:

        U-net model as specified by input arguments
    """
    if isinstance(in_spatial, int):
        d = in_spatial
        in_spatial = (None,) * d
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    if isinstance(filters, (tuple, list)):
        assert len(filters) == levels, f"List of filters has length {len(filters)} but u-net has {levels} levels."
    else:
        filters = (filters,) * levels
    # --- Construct the U-Net ---
    x = inputs = keras.Input(shape=in_spatial + (in_channels,))
    x = resnet_block(x.shape[-1], filters[0], periodic, batch_norm, activation, d)(x) if use_res_blocks else double_conv(x, d, filters[0], filters[0], batch_norm, activation, periodic)
    xs = [x]
    for i in range(1, levels):
        x = MAX_POOL[d](2, padding="same")(x)
        x = resnet_block(x.shape[-1], filters[i], periodic, batch_norm, activation, d)(x) if use_res_blocks else double_conv(x, d, filters[i], filters[i], batch_norm, activation, periodic)
        xs.insert(0, x)
    for i in range(1, levels):
        x = UPSAMPLE[d](2)(x)
        x = kl.Concatenate()([x, xs[i]])
        x = resnet_block(x.shape[-1], filters[i - 1], periodic, batch_norm, activation, d)(x) if use_res_blocks else double_conv(x, d, filters[i - 1], filters[i - 1], batch_norm, activation, periodic)
    x = CONV[d](out_channels, 1)(x)
    return keras.Model(inputs, x)


CONV = [None, kl.Conv1D, kl.Conv2D, kl.Conv3D]
MAX_POOL = [None, kl.MaxPool1D, kl.MaxPool2D, kl.MaxPool3D]
UPSAMPLE = [None, kl.UpSampling1D, kl.UpSampling2D, kl.UpSampling3D]
ACTIVATIONS = {'tanh': keras.activations.tanh, 'ReLU': keras.activations.relu, 'Sigmoid': keras.activations.sigmoid, 'SiLU': keras.activations.selu}


def pad_periodic(x: Tensor):
    d = len(x.shape) - 2
    if d >= 1:
        x = tf.concat([tf.expand_dims(x[:, -1, ...], axis=1), x, tf.expand_dims(x[:, 0, ...], axis=1)], axis=1)
    if d >= 2:
        x = tf.concat([tf.expand_dims(x[:, :, -1, ...], axis=2), x, tf.expand_dims(x[:, :, 0, ...], axis=2)], axis=2)
    if d >= 3:
        x = tf.concat([tf.expand_dims(x[:, :, :, -1, ...], axis=3), x, tf.expand_dims(x[:, :, :, 0, ...], axis=3)],
                      axis=3)
    return x


def double_conv(x, d: int, out_channels: int, mid_channels: int, batch_norm: bool, activation: Callable, periodic: bool):
    x = CONV[d](mid_channels, 3, padding='valid')(pad_periodic(x)) if periodic else CONV[d](mid_channels, 3, padding='same')(x)
    if batch_norm:
        x = kl.BatchNormalization()(x)
    x = activation(x)
    x = CONV[d](out_channels, 3, padding='valid')(pad_periodic(x)) if periodic else CONV[d](out_channels, 3, padding='same')(x)
    if batch_norm:
        x = kl.BatchNormalization()(x)
    x = activation(x)
    return x


def conv_net(in_channels: int,
             out_channels: int,
             layers: Tuple[int, ...] or List[int],
             batch_norm: bool = False,
             activation: str or Callable = 'ReLU',
             periodic=False,
             in_spatial: int or tuple = 2, **kwargs) -> keras.Model:
    """
    Built in Conv-Nets are also provided. Contrary to the classical convolutional neural networks, the feature map spatial size remains the same throughout the layers. Each layer of the network is essentially a convolutional block comprising of two conv layers. A filter size of 3 is used in the convolutional layers.
    Arguments:

        in_channels : input channels of the feature map, dtype : int
        out_channels : output channels of the feature map, dtype : int
        layers : list or tuple of output channels for each intermediate layer between the input and final output channels, dtype : list or tuple
        activation : activation function used within the layers, dtype : string
        batch_norm : use of batchnorm after each conv layer, dtype : bool
        in_spatial : spatial dimensions of the input feature map, dtype : int

    Returns:

        Conv-net model as specified by input arguments
    """
    if isinstance(in_spatial, int):
        d = in_spatial
        in_spatial = (None,) * d
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    x = inputs = keras.Input(shape=in_spatial + (in_channels,))
    if len(layers) < 1:
        layers.append(out_channels)
    for i in range(len(layers)):
        x = CONV[d](layers[i], 3, padding='valid')(pad_periodic(x)) if periodic else CONV[d](layers[i], 3, padding='same')(x)
        if batch_norm:
            x = kl.BatchNormalization()(x)
        x = activation(x)
    x = CONV[d](out_channels, 1)(x)
    return keras.Model(inputs, x)


def resnet_block(in_channels: int,
                 out_channels: int,
                 periodic: bool,
                 batch_norm: bool = False,
                 activation: str or Callable = 'ReLU',
                 in_spatial: int or tuple = 2):
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    if isinstance(in_spatial, int):
        d = in_spatial
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)
    x = x_1 = inputs = keras.Input(shape=(None,) * d + (in_channels,))
    x = CONV[d](out_channels, 3, padding='valid')(pad_periodic(x)) if periodic else CONV[d](out_channels, 3, padding='same')(x)
    if batch_norm:
        x = kl.BatchNormalization()(x)
    x = activation(x)
    x = CONV[d](out_channels, 3, padding='valid')(pad_periodic(x)) if periodic else CONV[d](out_channels, 3, padding='same')(x)
    if batch_norm:
        x = kl.BatchNormalization()(x)
    x = activation(x)
    if in_channels != out_channels:
        x_1 = CONV[d](out_channels, 1)(x_1)
        if batch_norm:
            x_1 = kl.BatchNormalization()(x_1)
    x = kl.Add()([x, x_1])
    return keras.Model(inputs, x)


def res_net(in_channels: int,
            out_channels: int,
            layers: Tuple[int, ...] or List[int],
            batch_norm: bool = False,
            activation: str or Callable = 'ReLU',
            periodic=False,
            in_spatial: int or tuple = 2, **kwargs):
    """
    Built in Res-Nets are provided in the ΦFlow framework. Similar to the conv-net, the feature map spatial size remains the same throughout the layers.
    These networks use residual blocks composed of two conv layers with a skip connection added from the input to the output feature map.
    A default filter size of 3 is used in the convolutional layers.

    Arguments:

        in_channels : input channels of the feature map, dtype : int
        out_channels : output channels of the feature map, dtype : int
        layers : list or tuple of output channels for each intermediate layer between the input and final output channels, dtype : list or tuple
        activation : activation function used within the layers, dtype : string
        batch_norm : use of batchnorm after each conv layer, dtype : bool
        in_spatial : spatial dimensions of the input feature map, dtype : int

    Returns:

        Res-net model as specified by input arguments
    """
    if isinstance(in_spatial, int):
        d = in_spatial
        in_spatial = (None,) * d
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)

    x = inputs = keras.Input(shape=in_spatial + (in_channels,))
    if len(layers) < 1:
        layers.append(out_channels)
    out = resnet_block(in_channels, layers[0], periodic, batch_norm, activation, d)(x)
    for i in range(1, len(layers)):
        out = resnet_block(layers[i - 1], layers[i], periodic, batch_norm, activation, d)(out)
    out = CONV[d](out_channels, 1)(out)
    return keras.Model(inputs, out)


def conv_classifier(in_features: int,
                    in_spatial: tuple or list,
                    num_classes: int,
                    blocks=(64, 128, 256, 256, 512, 512),
                    dense_layers=(4096, 4096, 100),
                    batch_norm=True,
                    activation='ReLU',
                    softmax=True,
                    periodic=False):
    """
    Based on VGG16.
    """
    assert isinstance(in_spatial, (tuple, list))
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    d = len(in_spatial)
    x = inputs = keras.Input(shape=in_spatial + (in_features,))
    for i, next in enumerate(blocks):
        if i in (0, 1):
            x = double_conv(x, d, next, next, batch_norm, activation, periodic)
            x = MAX_POOL[d](2)(x)
        else:
            x = double_conv(x, d, next, next, batch_norm, activation, periodic)
            x = CONV[d](next, 3, padding='valid')(pad_periodic(x)) if periodic else CONV[d](next, 3, padding='same')(x)
            if batch_norm:
                x = kl.BatchNormalization()(x)
            x = activation(x)
            x = MAX_POOL[d](2)(x)
    x = kl.Flatten()(x)
    flat_size = int(np.prod(in_spatial) * blocks[-1] / (2**d) ** len(blocks))
    x = dense_net(flat_size, num_classes, dense_layers, batch_norm, activation, softmax)(x)
    return keras.Model(inputs, x)


def get_mask(inputs, reverse_mask, data_format='NHWC'):
    """ Compute mask for slicing input feature map for Invertible Nets """
    shape = inputs.shape
    if len(shape) == 2:
        N = shape[-1]
        range_n = tf.range(0, N)
        even_ind = range_n % 2
        checker = tf.reshape(even_ind, (-1, N))
    elif len(shape) == 4:
        H = shape[2] if data_format == 'NCHW' else shape[1]
        W = shape[3] if data_format == 'NCHW' else shape[2]

        range_h = tf.range(0, H)
        range_w = tf.range(0, W)

        even_ind_h = tf.cast(range_h % 2, dtype=tf.bool)
        even_ind_w = tf.cast(range_w % 2, dtype=tf.bool)

        ind_h = tf.tile(tf.expand_dims(even_ind_h, -1), [1, W])
        ind_w = tf.tile(tf.expand_dims(even_ind_w, 0), [H, 1])
        # ind_h = even_ind_h.unsqueeze(-1).repeat(1, W)
        # ind_w = even_ind_w.unsqueeze( 0).repeat(H, 1)

        checker = tf.math.logical_xor(ind_h, ind_w)

        reshape = [-1, 1, H, W] if data_format == 'NCHW' else [-1, H, W, 1]
        checker = tf.reshape(checker, reshape)
        checker = tf.cast(checker, dtype=tf.float32)

    else:
        raise ValueError('Invalid tensor shape. Dimension of the tensor shape must be '
                         '2 (NxD) or 4 (NxCxHxW or NxHxWxC), got {}.'.format(inputs.get_shape().as_list()))

    if reverse_mask:
        checker = 1 - checker

    return checker


def Dense_resnet_block(in_channels: int,
                       mid_channels: int,
                       batch_norm: bool = False,
                       activation: str or Callable = 'ReLU'):
    inputs = keras.Input(shape=(in_channels,))
    x_1 = inputs
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation

    x = kl.Dense(mid_channels)(inputs)
    if batch_norm:
        x = kl.BatchNormalization()(x)
    x = activation(x)

    x = kl.Dense(in_channels)(x)
    if batch_norm:
        x = kl.BatchNormalization()(x)
    x = activation(x)

    x = kl.Add()([x, x_1])

    return keras.Model(inputs, x)


NET = {'u_net': u_net, 'res_net': res_net, 'conv_net': conv_net}


class CouplingLayer(keras.Model):

    def __init__(self, in_channels, activation, batch_norm, in_spatial, net, reverse_mask):
        super(CouplingLayer, self).__init__()

        self.activation = activation
        self.batch_norm = batch_norm
        self.reverse_mask = reverse_mask

        if in_spatial == 0:  # for in_spatial = 0, use dense layers
            self.s1 = Dense_resnet_block(in_channels, in_channels, batch_norm, activation)
            self.t1 = Dense_resnet_block(in_channels, in_channels, batch_norm, activation)

            self.s2 = Dense_resnet_block(in_channels, in_channels, batch_norm, activation)
            self.t2 = Dense_resnet_block(in_channels, in_channels, batch_norm, activation)
        else:
            self.s1 = NET[net](in_channels=in_channels, out_channels=in_channels, layers=[],
                               batch_norm=batch_norm, activation=activation,
                               in_spatial=in_spatial)
            self.t1 = NET[net](in_channels=in_channels, out_channels=in_channels, layers=[],
                               batch_norm=batch_norm, activation=activation,
                               in_spatial=in_spatial)

            self.s2 = NET[net](in_channels=in_channels, out_channels=in_channels, layers=[],
                               batch_norm=batch_norm, activation=activation,
                               in_spatial=in_spatial)
            self.t2 = NET[net](in_channels=in_channels, out_channels=in_channels, layers=[],
                               batch_norm=batch_norm, activation=activation,
                               in_spatial=in_spatial)

    def call(self, x, invert=False):
        mask = get_mask(x, self.reverse_mask, 'NCHW')

        if invert:
            v1 = x * mask
            v2 = x * (1 - mask)

            u2 = (1 - mask) * (v2 - self.t1(v1)) * tf.math.exp(tf.tanh(-self.s1(v1)))
            u1 = mask * (v1 - self.t2(u2)) * tf.math.exp(tf.tanh(-self.s2(u2)))

            return u1 + u2
        else:
            u1 = x * mask
            u2 = x * (1 - mask)

            v1 = mask * (u1 * tf.math.exp(tf.tanh(self.s2(u2))) + self.t2(u2))
            v2 = (1 - mask) * (u2 * tf.math.exp(tf.tanh(self.s1(v1))) + self.t1(v1))

            return v1 + v2


class InvertibleNet(keras.Model):
    def __init__(self, in_channels, num_blocks, activation, batch_norm, in_spatial, net):
        super(InvertibleNet, self).__init__()
        self.num_blocks = num_blocks

        self.layer_dict = {}
        for i in range(num_blocks):
            self.layer_dict[f'coupling_block{i + 1}'] = \
                CouplingLayer(in_channels,
                              activation, batch_norm,
                              in_spatial, net, (i % 2 == 0))

    def call(self, x, backward=False):
        if backward:
            for i in range(self.num_blocks, 0, -1):
                x = self.layer_dict[f'coupling_block{i}'](x, backward)
        else:
            for i in range(1, self.num_blocks + 1):
                x = self.layer_dict[f'coupling_block{i}'](x)
        return x


def invertible_net(in_channels: int,
                   num_blocks: int,
                   batch_norm: bool = False,
                   net: str = 'u_net',
                   activation: str or type = 'ReLU',
                   in_spatial: tuple or int = 2, **kwargs):
    """
    ΦFlow also provides invertible neural networks that are capable of inverting the output tensor back to the input tensor initially passed.\ These networks have far reaching applications in predicting input parameters of a problem given its observations.\ Invertible nets are composed of multiple concatenated coupling blocks wherein each such block consists of arbitrary neural networks.

    Currently, these arbitrary neural networks could be set to u_net(default), conv_net, res_net or dense_net blocks with in_channels = out_channels.
    The architecture used is popularized by ["Real NVP"](https://arxiv.org/abs/1605.08803).

    Arguments:

        in_channels : input channels of the feature map, dtype : int
        num_blocks : number of coupling blocks inside the invertible net, dtype : int
        activation : activation function used within the layers, dtype : string
        batch_norm : use of batchnorm after each layer, dtype : bool
        in_spatial : spatial dimensions of the input feature map, dtype : int
        net : type of neural network blocks used in coupling layers, dtype : str
        **kwargs : placeholder for arguments not supported by the function

    Returns:

        Invertible Net model as specified by input arguments

    Note: Currently supported values for net are 'u_net'(default), 'conv_net' and 'res_net'.
    For choosing 'dense_net' as the network block in coupling layers in_spatial must be set to zero.
    """
    if isinstance(in_spatial, tuple):
        in_spatial = len(in_spatial)

    return InvertibleNet(in_channels, num_blocks, activation,
                         batch_norm, in_spatial, net)


##################################################################################################################
#  Fourier Neural Operators
#  source: https://github.com/zongyi-li/fourier_neural_operator
###################################################################################################################
RFFT = [None, tf.signal.rfft, tf.signal.rfft2d, tf.signal.rfft3d]
FFT = [None, tf.signal.fft, tf.signal.fft2d, tf.signal.fft3d]
IRFFT = [None, tf.signal.irfft, tf.signal.irfft2d, tf.signal.irfft3d]

class SpectralConv(keras.Model):

    def __init__(self, in_channels, out_channels, modes, in_spatial):

        super(SpectralConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_spatial = in_spatial
        assert in_spatial >= 1 and in_spatial <= 3
        if isinstance(modes, int):
            mode = modes
            modes = [mode for i in range(in_spatial)]

        self.scale = 1 / (in_channels * out_channels)

        self.modes = {i + 1: modes[i] for i in range(len(modes))}
        self.weights_ = {}

        rand_shape = [in_channels, out_channels]
        rand_shape += [self.modes[i] for i in range(1, in_spatial + 1)]

        for i in range(2 ** (in_spatial - 1)):
            self.weights_[f'w{i + 1}'] = tf.complex(tf.Variable(self.scale * tf.random.normal(shape=rand_shape, dtype=tf.dtypes.float32), trainable=True),
                                                tf.Variable(self.scale * tf.random.normal(shape=rand_shape, dtype=tf.dtypes.float32), trainable=True))

    def complex_mul(self, input, weights):

        if self.in_spatial == 1:
            return tf.einsum("bix,iox->box", input, weights)
        elif self.in_spatial == 2:
            return tf.einsum("bixy,ioxy->boxy", input, weights)
        elif self.in_spatial == 3:
            return tf.einsum("bixyz,ioxyz->boxyz", input, weights)


    def call(self, x):
        batch_size = x.shape[0]

        x_ft = RFFT[self.in_spatial](x)

        outft_dims = [batch_size, self.out_channels] + \
                     [x.shape[-i] for i in range(self.in_spatial, 1, -1)] + [x.shape[-1] // 2 + 1]
        out_ft0 = tf.complex(tf.Variable(tf.zeros(outft_dims, dtype=tf.dtypes.float32)),
                            tf.Variable(tf.zeros(outft_dims, dtype=tf.dtypes.float32)))

        if self.in_spatial == 1:
            out_ft1 = self.complex_mul(x_ft[:, :, :self.modes[1]],
                                      self.weights_['w1'])
            out_ft = tf.concat([out_ft1, out_ft0[:, :, self.modes[1]:]], axis=-1)
        elif self.in_spatial == 2:
            out_ft1 = self.complex_mul(x_ft[:, :, :self.modes[1], :self.modes[2]],
                                 self.weights_['w1'])
            out_ft2 = self.complex_mul(x_ft[:, :, -self.modes[1]:, :self.modes[2]],
                                 self.weights_['w2'])
            out_ft3 = tf.concat([out_ft1, out_ft0[:, :, self.modes[1]:-self.modes[1],
                                         :self.modes[2]], out_ft2], axis=-2)
            out_ft = tf.concat([out_ft3, out_ft0[:, :, :, self.modes[2]:]], axis=-1)
        elif self.in_spatial == 3:
            out_ft1 = self.complex_mul(x_ft[:, :, :self.modes[1], :self.modes[2], :self.modes[3]],
                                 self.weights_['w1'])
            out_ft2 = self.complex_mul(x_ft[:, :, -self.modes[1]:, :self.modes[2], :self.modes[3]],
                                 self.weights_['w2'])
            out_ft3 = self.complex_mul(x_ft[:, :, :self.modes[1], -self.modes[2]:, :self.modes[3]],
                                 self.weights_['w3'])
            out_ft4 = self.complex_mul(x_ft[:, :, -self.modes[1]:, -self.modes[2]:, :self.modes[3]],
                                 self.weights_['w4'])

            out_ft5 = tf.concat([out_ft1, out_ft0[:, :, self.modes[1]:-self.modes[1], :self.modes[2], :self.modes[3]]
                                    , out_ft2], axis=-3)
            out_ft6 = tf.concat([out_ft3, out_ft0[:, :, self.modes[1]:-self.modes[1], -self.modes[2]:, :self.modes[3]]
                                    , out_ft4], axis=-3)
            out_ft7 = tf.concat([out_ft5, out_ft0[:, :, :, self.modes[2]:-self.modes[2], :self.modes[3]], out_ft6],
                                axis=-2)
            out_ft = tf.concat([out_ft7, out_ft0[:, :, :, :, self.modes[3]:]], axis=-1)

        ##Return to Physical Space
        x = IRFFT[self.in_spatial](out_ft)

        return x


class FNO(keras.Model):

    def __init__(self, in_channels, out_channels, width, modes, activation, batch_norm, in_spatial):
        super(FNO, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2.

        input shape and output shape: (batchsize b, channels c, *spatial)
        """

        self.activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
        self.width = width
        self.in_spatial = in_spatial

        self.fc0 = kl.Dense(self.width)

        self.model_dict = {}
        for i in range(4):
            self.model_dict[f'conv{i}'] = SpectralConv(self.width, self.width, modes, in_spatial)
            self.model_dict[f'w{i}'] = CONV[self.in_spatial](self.width, kernel_size=1)
            self.model_dict[f'bn{i}'] = kl.BatchNormalization()

        self.fc1 = kl.Dense(128)
        self.fc2 = kl.Dense(out_channels)

    # Adding extra spatial channels eg. x, y, z, .... to input x
    def get_grid(self, shape, device):
        batch_size = shape[0]
        grid_channel_sizes = shape[1:-1]  # shape =  (batch_size, *spatial, channels)
        self.grid_channels = {}
        for i in range(self.in_spatial):
            self.grid_channels[f'dim{i}'] = tf.cast(tf.linspace(0, 1,
                                        grid_channel_sizes[i]), dtype=tf.dtypes.float32)  #tf.tensor(tf.linspace(0, 1, grid_channel_sizes[i]), dtype=tf.dtypes.float32)
            reshape_dim_tuple = [1,] + [1 if i != j else grid_channel_sizes[j]
                                        for j in range(self.in_spatial)] + [1,]
            repeat_dim_tuple = [batch_size,] + [1 if i == j else grid_channel_sizes[j]
                                                for j in range(self.in_spatial)] + [1,]

            self.grid_channels[f'dim{i}'] = tf.tile(tf.reshape(self.grid_channels[f'dim{i}'], reshape_dim_tuple), repeat_dim_tuple)

        return tf.concat([self.grid_channels[f'dim{i}'] for i in range(self.in_spatial)], axis=-1)

    def call(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = tf.concat([x, grid], axis=-1)

        permute_tuple= [0] + [self.in_spatial + 1] + [i + 1 for i in range(self.in_spatial)]
        permute_tuple_reverse = [0] + [2 + i for i in range(self.in_spatial)] + [1]

        # No need to Transpose x such that channels shape lies
        # at the end to pass it through linear layers as it's the default in tf
        #x = tf.transpose(x, permute_tuple)

        x = self.fc0(x)

        for i in range(4):
            x1 = self.model_dict[f'w{i}'](x)
            # Spectral conv expects a shape : [batch, channel, *spatial]
            # hence the transpose:
            x2 = self.model_dict[f'conv{i}'](tf.transpose(x, permute_tuple))
            x2 = tf.transpose(x2, permute_tuple_reverse)
            x = self.model_dict[f'bn{i}'](x1) + self.model_dict[f'bn{i}'](x2)
            x = self.activation(x)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x


def fno(in_channels: int,
        out_channels: int,
        mid_channels: int,
        modes: Tuple[int, ...] or List[int],
        activation: str or type = 'ReLU',
        batch_norm: bool = False,
        in_spatial: int = 2):
    """
    ["Fourier Neural Operator"](https://github.com/zongyi-li/fourier_neural_operator) network contains 4 layers of the Fourier layer.
    1. Lift the input to the desire channel dimension by self.fc0 .
    2. 4 layers of the integral operators u' = (W + K)(u). W defined by self.w; K defined by self.conv .
    3. Project from the channel space to the output space by self.fc1 and self.fc2.

    Arguments:

        in_channels : input channels of the feature map, dtype : int
        out_channels : output channels of the feature map, dtype : int
        mid_channels : channels used in Spectral Convolution Layers, dtype : int
        modes : Fourier modes for each spatial channel, dtype : List[int] or int (in case all number modes are to be the same for each spatial channel)
        activation : activation function used within the layers, dtype : string
        batch_norm : use of batchnorm after each conv layer, dtype : bool
        in_spatial : spatial dimensions of the input feature map, dtype : int

    Returns:

        Fourier Neural Operator model as specified by input arguments.
    """
    net = FNO(in_channels, out_channels, mid_channels, modes, activation, batch_norm, in_spatial)
    return net