import numpy
from tensorflow import keras
from tensorflow.keras import layers as kl


def parameter_count(model: keras.Model):
    total = 0
    for parameter in model.trainable_weights:
        total += numpy.prod(parameter.shape)
    return int(total)


def dense_net(in_channels: int,
              out_channels: int,
              layers: tuple or list,
              activation='ReLU') -> keras.Model:
    if isinstance(activation, str):
        activation = {'tanh': keras.activations.tanh,
                      'ReLU': keras.activations.relu}[activation]
    dense_layers = [kl.Dense(l, activation=activation) for l in layers]
    return keras.models.Sequential([kl.InputLayer(input_shape=(in_channels,)), *dense_layers, kl.Dense(out_channels, activation='linear')])


def u_net(in_channels: int,
          out_channels: int,
          levels: int = 4,
          filters: int or tuple or list = 16,
          batch_norm=True,
          in_spatial: tuple = None) -> keras.Model:
    if not batch_norm:
        raise NotImplementedError("only batch_norm=True currently supported")
    if in_spatial is None:
        in_spatial = (None, None)
    if isinstance(filters, (tuple, list)):
        assert len(filters) == levels, f"List of filters has length {len(filters)} but u-net has {levels} levels."
    else:
        filters = (filters,) * levels

    inputs = keras.Input(shape=in_spatial + (in_channels,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = kl.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = kl.BatchNormalization()(x)
    x = kl.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filter_count in filters:  # [64, 128, 256]:
        x = kl.Activation("relu")(x)
        x = kl.SeparableConv2D(filter_count, 3, padding="same")(x)
        x = kl.BatchNormalization()(x)

        x = kl.Activation("relu")(x)
        x = kl.SeparableConv2D(filter_count, 3, padding="same")(x)
        x = kl.BatchNormalization()(x)

        x = kl.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = kl.Conv2D(filter_count, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = kl.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filter_count in tuple(reversed(filters)) + (filters[-1],):  # [256, 128, 64, 32]:
        x = kl.Activation("relu")(x)
        x = kl.Conv2DTranspose(filter_count, 3, padding="same")(x)
        x = kl.BatchNormalization()(x)

        x = kl.Activation("relu")(x)
        x = kl.Conv2DTranspose(filter_count, 3, padding="same")(x)
        x = kl.BatchNormalization()(x)

        x = kl.UpSampling2D(2)(x)

        # Project residual
        residual = kl.UpSampling2D(2)(previous_block_activation)
        residual = kl.Conv2D(filter_count, 1, padding="same")(residual)
        x = kl.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = kl.Conv2D(out_channels, 3, activation=None, padding="same")(x)
    return keras.Model(inputs, outputs)
