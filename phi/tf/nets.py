import numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as kl
import pickle

from typing import Callable


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
    Analogue functions exist for other learning frameworks.
    """
    return keras.optimizers.Adam(learning_rate, betas[0], betas[1], epsilon)


def dense_net(in_channels: int,
              out_channels: int,
              layers: tuple or list,
              batch_norm=False,
              activation='ReLU') -> keras.Model:
    activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    keras_layers = []
    for neuron_count in layers:
        keras_layers.append(kl.Dense(neuron_count, activation=activation))
        if batch_norm:
            keras_layers.append(kl.BatchNormalization())
    return keras.models.Sequential([kl.InputLayer(input_shape=(in_channels,)), *keras_layers, kl.Dense(out_channels, activation='linear')])


def u_net(in_channels: int,
          out_channels: int,
          levels: int = 4,
          filters: int or tuple or list = 16,
          batch_norm: bool = True,
          activation: str or Callable = 'ReLU',
          in_spatial: tuple or int = 2) -> keras.Model:
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
    x = double_conv(x, d, filters[0], filters[0], batch_norm, activation)
    xs = [x]
    for i in range(1, levels):
        x = MAX_POOL[d](2, padding="same")(x)
        x = double_conv(x, d, filters[i], filters[i], batch_norm, activation)
        xs.insert(0, x)
    for i in range(1, levels):
        x = UPSAMPLE[d](2)(x)
        x = kl.Concatenate()([x, xs[i]])
        x = double_conv(x, d, filters[i - 1], filters[i - 1], batch_norm, activation)
    x = CONV[d](out_channels, 1)(x)
    return keras.Model(inputs, x)


CONV = [None, kl.Conv1D, kl.Conv2D, kl.Conv3D]
MAX_POOL = [None, kl.MaxPool1D, kl.MaxPool2D, kl.MaxPool3D]
UPSAMPLE = [None, kl.UpSampling1D, kl.UpSampling2D, kl.UpSampling3D]
ACTIVATIONS = {'tanh': keras.activations.tanh, 'ReLU': keras.activations.relu, 'Sigmoid': keras.activations.sigmoid}


def double_conv(x, d: int, out_channels: int, mid_channels: int, batch_norm: bool, activation: Callable):
    x = CONV[d](mid_channels, 3, padding='same')(x)
    if batch_norm:
        x = kl.BatchNormalization()(x)
    x = activation(x)
    x = CONV[d](out_channels, 3, padding='same')(x)
    if batch_norm:
        x = kl.BatchNormalization()(x)
    x = activation(x)
    return x
