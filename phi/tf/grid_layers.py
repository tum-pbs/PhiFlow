from phi.geom import AABox
from phi.physics.field import CenteredGrid
from . import tf


def conv_layer(grid, filters, kernel_size, strides=1, padding='valid', activation=None, use_bias=True, name=None, trainable=True, reuse=None):
    assert isinstance(grid, CenteredGrid)
    if grid.rank == 2:
        result = tf.layers.conv2d(grid.data, filters, kernel_size, strides=strides, activation=activation, padding=padding, name=name, use_bias=use_bias, trainable=trainable, reuse=reuse)
    elif grid.rank == 1:
        result = tf.layers.conv1d(grid.data, filters, kernel_size, strides=strides, activation=activation, padding=padding, name=name, use_bias=use_bias, trainable=trainable, reuse=reuse)
    else:
        raise NotImplementedError()
    if padding == 'valid':
        w_upper = kernel_size // 2
        w_lower = (kernel_size - 1) // 2
        box = AABox(grid.box.lower + w_lower * grid.dx, grid.box.upper - w_upper * grid.dx)
    else:
        box = grid.box
    return CenteredGrid(result, box=box, extrapolation=grid.extrapolation)


def residual_block(grid, nb_channels, kernel_size=(3, 3), _strides=(1, 1), activation=tf.nn.leaky_relu, _project_shortcut=False, padding="SYMMETRIC", name=None, training=False, trainable=True, reuse=tf.AUTO_REUSE):
    y = grid.data
    shortcut = y

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    pad1 = [(kernel_size[0] - 1) // 2, kernel_size[0] // 2]
    pad2 = [(kernel_size[1] - 1) // 2, kernel_size[1] // 2]

    # down-sampling is performed with a stride of 2
    y = tf.pad(y, [[0, 0], pad1, pad2, [0, 0]], mode=padding)
    y = tf.layers.conv2d(y, nb_channels, kernel_size=kernel_size, strides=_strides, padding='valid', name=None if name is None else name + "/conv1", trainable=trainable, reuse=reuse)
    # y = tf.layers.batch_normalization(y, name=None if name is None else name+"/norm1", training=training, trainable=trainable, reuse=reuse)
    y = activation(y)

    y = tf.pad(y, [[0, 0], pad1, pad2, [0, 0]], mode=padding)
    y = tf.layers.conv2d(y, nb_channels, kernel_size=kernel_size, strides=(1, 1), padding='valid', name=None if name is None else name + "/conv2", trainable=trainable, reuse=reuse)
    # y = tf.layers.batch_normalization(y, name=None if name is None else name+"/norm2", training=training, trainable=trainable, reuse=reuse)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = tf.pad(shortcut, [[0, 0], pad1, pad2, [0, 0]], mode=padding)
        shortcut = tf.layers.conv2d(shortcut, nb_channels, kernel_size=(1, 1), strides=_strides, padding='valid', name=None if name is None else name + "/convid", trainable=trainable, reuse=reuse)
        # shortcut = tf.layers.batch_normalization(shortcut, name=None if name is None else name+"/normid", training=training, trainable=trainable, reuse=reuse)

    y += shortcut
    y = activation(y)
    return CenteredGrid(y, box=grid.box, extrapolation=grid.extrapolation)
