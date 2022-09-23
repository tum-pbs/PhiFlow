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
                phi_tensor = math.wrap(var, math.channel('input,output'))
            elif name.endswith('.bias'):
                phi_tensor = math.wrap(var, math.channel('output'))
            else:
                raise NotImplementedError(name)
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
              activation='ReLU') -> keras.Model:
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    keras_layers = []
    for neuron_count in layers:
        keras_layers.append(kl.Dense(neuron_count, activation=activation))
        if batch_norm:
            keras_layers.append(kl.BatchNormalization())
    return keras.models.Sequential([kl.InputLayer(input_shape=(in_channels,)),
                                    *keras_layers,
                                    kl.Dense(out_channels, activation='linear')])


def u_net(in_channels: int,
          out_channels: int,
          levels: int = 4,
          filters: int or tuple or list = 16,
          batch_norm: bool = True,
          activation: str or Callable = 'ReLU',
          in_spatial: tuple or int = 2,
          use_res_blocks: bool = False, **kwargs) -> keras.Model:
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
    x = resnet_block(x.shape[-1], filters[0], batch_norm, activation, d)(x) if use_res_blocks else double_conv(x, d, filters[0], filters[0], batch_norm, activation)
    xs = [x]
    for i in range(1, levels):
        x = MAX_POOL[d](2, padding="same")(x)
        x = resnet_block(x.shape[-1], filters[i], batch_norm, activation, d)(x) if use_res_blocks else double_conv(x, d, filters[i], filters[i], batch_norm, activation)
        xs.insert(0, x)
    for i in range(1, levels):
        x = UPSAMPLE[d](2)(x)
        x = kl.Concatenate()([x, xs[i]])
        x = resnet_block(x.shape[-1], filters[i - 1], batch_norm, activation, d)(x) if use_res_blocks else double_conv(x, d, filters[i - 1], filters[i - 1], batch_norm, activation)
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


def double_conv(x, d: int, out_channels: int, mid_channels: int, batch_norm: bool, activation: Callable):
    x = pad_periodic(x)
    x = CONV[d](mid_channels, 3, padding='valid')(x)
    # x = CONV[d](mid_channels, 3, padding='same')(x)
    if batch_norm:
        x = kl.BatchNormalization()(x)
    x = activation(x)

    x = pad_periodic(x)
    x = CONV[d](out_channels, 3, padding='valid')(x)
    # x = CONV[d](out_channels, 3, padding='same')(x)
    if batch_norm:
        x = kl.BatchNormalization()(x)
    x = activation(x)
    return x


def conv_net(in_channels: int,
             out_channels: int,
             layers: Tuple[int, ...] or List[int],
             batch_norm: bool = False,
             activation: str or Callable = 'ReLU',
             in_spatial: int or tuple = 2, **kwargs) -> keras.Model:
    if isinstance(in_spatial, int):
        d = (None,) * in_spatial
    else:
        assert isinstance(in_spatial, tuple)
        d = in_spatial
        in_spatial = len(d)
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    x = inputs = keras.Input(shape=d + (in_channels,))
    if len(layers) < 1:
        layers.append(out_channels)
    for i in range(len(layers)):
        x = pad_periodic(x)
        x = CONV[in_spatial](layers[i], 3, padding='valid')(x)
        if batch_norm:
            x = kl.BatchNormalization()(x)
        x = activation(x)
    # x = pad_periodic(x)
    x = CONV[in_spatial](out_channels, 1)(x)
    return keras.Model(inputs, x)


def resnet_block(in_channels: int,
                 out_channels: int,
                 batch_norm: bool = False,
                 activation: str or Callable = 'ReLU',
                 in_spatial: int or tuple = 2):
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    if isinstance(in_spatial, int):
        d = (None,) * in_spatial
    else:
        assert isinstance(in_spatial, tuple)
        d = in_spatial
        in_spatial = len(d)
    d = (None,) * in_spatial
    inputs = keras.Input(shape=d + (in_channels,))
    x_1 = inputs
    x = pad_periodic(inputs)
    x = CONV[in_spatial](out_channels, 3, padding='valid')(x)
    if batch_norm:
        x = kl.BatchNormalization()(x)
    x = activation(x)
    x = pad_periodic(x)
    x = CONV[in_spatial](out_channels, 3, padding='valid')(x)
    if batch_norm:
        x = kl.BatchNormalization()(x)
    x = activation(x)
    if in_channels != out_channels:
        x_1 = CONV[in_spatial](out_channels, 1)(x_1)
        if batch_norm:
            x_1 = kl.BatchNormalization()(x_1)
    x = kl.Add()([x, x_1])
    return keras.Model(inputs, x)


def res_net(in_channels: int,
            out_channels: int,
            layers: Tuple[int, ...] or List[int],
            batch_norm: bool = False,
            activation: str or Callable = 'ReLU',
            in_spatial: int or tuple = 2, **kwargs):
    if isinstance(in_spatial, int):
        d = (None,) * in_spatial
    else:
        assert isinstance(in_spatial, tuple)
        d = in_spatial
        in_spatial = len(d)

    x = inputs = keras.Input(shape=d + (in_channels,))
    if len(layers) < 1:
        layers.append(out_channels)
    out = resnet_block(in_channels, layers[0], batch_norm, activation, in_spatial)(x)
    for i in range(1, len(layers)):
        out = resnet_block(layers[i - 1], layers[i], batch_norm, activation, in_spatial)(out)
    out = CONV[in_spatial](out_channels, 1)(out)
    return keras.Model(inputs, out)


def conv_classifier(input_shape: list, num_classes: int, batch_norm: bool, in_spatial: int or tuple):
    if isinstance(in_spatial, int):
        d = in_spatial
        in_spatial = (None,) * d
    else:
        assert isinstance(in_spatial, tuple)
        d = len(in_spatial)
    # input_shape[0] = Channels
    spatial_shape_list = list(input_shape[1:])
    x = inputs = keras.Input(shape=in_spatial + (input_shape[0],))
    x = double_conv(x, d, 64, 64, batch_norm, ACTIVATIONS['ReLU'])
    x = MAX_POOL[d](2)(x)

    x = double_conv(x, d, 128, 128, batch_norm, ACTIVATIONS['ReLU'])
    x = MAX_POOL[d](2)(x)

    x = double_conv(x, d, 256, 256, batch_norm, ACTIVATIONS['ReLU'])
    x = pad_periodic(x)
    x = CONV[d](256, 3, padding='valid')(x)
    if batch_norm:
        x = kl.BatchNormalization()(x)
    x = ACTIVATIONS['ReLU'](x)
    x = MAX_POOL[d](2)(x)

    x = double_conv(x, d, 512, 512, batch_norm, ACTIVATIONS['ReLU'])
    x = pad_periodic(x)
    x = CONV[d](512, 3, padding='valid')(x)
    if batch_norm:
        x = kl.BatchNormalization()(x)
    x = ACTIVATIONS['ReLU'](x)
    x = MAX_POOL[d](2)(x)

    x = double_conv(x, d, 512, 512, batch_norm, ACTIVATIONS['ReLU'])
    x = pad_periodic(x)
    x = CONV[d](512, 3, padding='valid')(x)
    if batch_norm:
        x = kl.BatchNormalization()(x)
    x = ACTIVATIONS['ReLU'](x)
    x = MAX_POOL[d](2)(x)

    for i in range(5):
        for j in range(len(spatial_shape_list)):
            spatial_shape_list[j] = math.floor((spatial_shape_list[j] - 2) / 2) + 1

    flattened_input_dim = 1
    for i in range(len(spatial_shape_list)):
        flattened_input_dim *= spatial_shape_list[i]
    flattened_input_dim *= 512

    x = kl.Flatten()(x)
    x = dense_net(flattened_input_dim, num_classes, [4096, 4096, 100], batch_norm, 'ReLU')(x)

    x = kl.Softmax()(x)

    return keras.Model(inputs, x)


def get_mask(inputs, reverse_mask, data_format='NHWC'):
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
    if isinstance(in_spatial, tuple):
        in_spatial = len(in_spatial)

    return InvertibleNet(in_channels, num_blocks, activation,
                         batch_norm, in_spatial, net)
