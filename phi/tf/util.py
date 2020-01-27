# coding=utf-8
import logging
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

from phi import struct
from phi.math.math_util import is_static_shape
from phi.physics.field.staggered_grid import StaggeredGrid
from phi.physics.field.grid import CenteredGrid

if tf.__version__[0] == '2':
    logging.info('Adjusting for tensorflow 2.0')
    tf = tf.compat.v1
    tf.disable_eager_execution()


def _tf_name(trace, basename):
    if basename is None:
        return trace.path('/')
    else:
        return basename + '/' + trace.path('/')


def placeholder(shape, dtype=np.float32, basename=None, item_condition=struct.VARIABLES):
    if struct.isstruct(dtype):
        def placeholder_map(trace):
            shape, dtype = trace.value
            return tf.placeholder(dtype, shape, _tf_name(trace, basename))
        zipped = struct.zip([shape, dtype], leaf_condition=is_static_shape, item_condition=item_condition)
        return struct.map(placeholder_map, zipped, leaf_condition=is_static_shape, trace=True, item_condition=item_condition)
    else:
        def f(trace): return tf.placeholder(dtype, trace.value, _tf_name(trace, basename))
        return struct.map(f, shape, leaf_condition=is_static_shape, trace=True, item_condition=item_condition)


def placeholder_like(obj, basename=None):
    warnings.warn("placeholder_like may not respect the batch dimension. "
                  "For State objects, use placeholder(state.shape) instead.", DeprecationWarning, stacklevel=2)

    def f(attr): return tf.placeholder(attr.value.dtype, attr.value.shape, _tf_name(attr, basename))
    return struct.map(f, obj, leaf_condition=is_static_shape, trace=True)


def variable(initial_value, dtype=np.float32, basename=None, trainable=True, item_condition=struct.VARIABLES):
    def f(attr): return tf.Variable(attr.value, name=_tf_name(attr, basename), dtype=dtype, trainable=trainable)
    return struct.map(f, initial_value, trace=True, item_condition=item_condition)


def variable_generator(initializer, dtype=np.float32, basename=None, trainable=True):
    def create_variable(shape):
        initial_value = initializer(shape)
        return variable(initial_value, dtype, basename, trainable)
    return create_variable


def isplaceholder(obj):
    return isinstance(obj, tf.Tensor) and obj.op.type == 'Placeholder'


def group_normalization(x, group_count, eps=1e-5):
    batch_size, H, W, C = tf.shape(x)
    gamma = tf.Variable(np.ones([1, 1, 1, C]), dtype=tf.float32, name="GN_gamma")
    beta = tf.Variable(np.zeros([1, 1, 1, C]), dtype=tf.float32, name="GN_beta")
    x = tf.reshape(x, [batch_size, group_count, H, W, C // group_count])
    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + eps)
    x = tf.reshape(x, [batch_size, H, W, C])
    return x * gamma + beta


def residual_block(y, nb_channels, kernel_size=(3, 3), _strides=(1, 1), activation=tf.nn.leaky_relu,
                   _project_shortcut=False, padding="SYMMETRIC", name=None, training=False, trainable=True, reuse=tf.AUTO_REUSE):
    shortcut = y

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    pad1 = [(kernel_size[0] - 1) // 2, kernel_size[0] // 2]
    pad2 = [(kernel_size[1] - 1) // 2, kernel_size[1] // 2]

    # down-sampling is performed with a stride of 2
    y = tf.pad(y, [[0, 0], pad1, pad2, [0, 0]], mode=padding)
    y = tf.layers.conv2d(y, nb_channels, kernel_size=kernel_size, strides=_strides, padding='valid',
                         name=None if name is None else name + "/conv1", trainable=trainable, reuse=reuse)
    # y = tf.layers.batch_normalization(y, name=None if name is None else name+"/norm1", training=training, trainable=trainable, reuse=reuse)
    y = activation(y)

    y = tf.pad(y, [[0, 0], pad1, pad2, [0, 0]], mode=padding)
    y = tf.layers.conv2d(y, nb_channels, kernel_size=kernel_size, strides=(1, 1), padding='valid',
                         name=None if name is None else name + "/conv2", trainable=trainable, reuse=reuse)
    # y = tf.layers.batch_normalization(y, name=None if name is None else name+"/norm2", training=training, trainable=trainable, reuse=reuse)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = tf.pad(shortcut, [[0, 0], pad1, pad2, [0, 0]], mode=padding)
        shortcut = tf.layers.conv2d(shortcut, nb_channels, kernel_size=(1, 1), strides=_strides, padding='valid',
                                    name=None if name is None else name + "/convid", trainable=trainable, reuse=reuse)
        # shortcut = tf.layers.batch_normalization(shortcut, name=None if name is None else name+"/normid", training=training, trainable=trainable, reuse=reuse)

    y += shortcut
    y = activation(y)

    return y


def residual_block_1d(y, nb_channels, kernel_size=(3,), _strides=(1,), activation=tf.nn.leaky_relu,
                      _project_shortcut=False, padding="SYMMETRIC", name=None, training=False, trainable=True, reuse=tf.AUTO_REUSE):
    shortcut = y

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)

    pad1 = [(kernel_size[0] - 1) // 2, kernel_size[0] // 2]

    # down-sampling is performed with a stride of 2
    y = tf.pad(y, [[0, 0], pad1, [0, 0]], mode=padding)
    y = tf.layers.conv1d(y, nb_channels, kernel_size=kernel_size, strides=_strides, padding='valid',
                         name=None if name is None else name + "/conv1", trainable=trainable, reuse=reuse)
    # y = tf.layers.batch_normalization(y, name=None if name is None else name+"/norm1", training=training, trainable=trainable, reuse=reuse)
    y = activation(y)

    y = tf.pad(y, [[0, 0], pad1, [0, 0]], mode=padding)
    y = tf.layers.conv1d(y, nb_channels, kernel_size=kernel_size, strides=(1,), padding='valid',
                         name=None if name is None else name + "/conv2", trainable=trainable, reuse=reuse)
    # y = tf.layers.batch_normalization(y, name=None if name is None else name+"/norm2", training=training, trainable=trainable, reuse=reuse)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1,):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = tf.pad(shortcut, [[0, 0], pad1, [0, 0]], mode=padding)
        shortcut = tf.layers.conv1d(shortcut, nb_channels, kernel_size=(1, 1), strides=_strides, padding='valid',
                                    name=None if name is None else name + "/convid", trainable=trainable, reuse=reuse)
        # shortcut = tf.layers.batch_normalization(shortcut, name=None if name is None else name+"/normid", training=training, trainable=trainable, reuse=reuse)

    y += shortcut
    y = activation(y)

    return y


def istensor(obj):
    if isinstance(obj, CenteredGrid):
        return istensor(obj.data)
    if isinstance(obj, StaggeredGrid):
        return np.any([istensor(t) for t in obj.data])
    return isinstance(obj, (tf.Tensor, tf.Variable))


def conv_function(scope, constants_file=None):
    if constants_file is not None:
        reader = pywrap_tensorflow.NewCheckpointReader(constants_file)

        def conv(n, filters, kernel_size, strides=[1, 1, 1, 1], padding="VALID", activation=None, name=None, kernel_initializer=None):
            assert name is not None
            kernel = reader.get_tensor("%s/%s/kernel" % (scope, name))
            assert kernel.shape[-1] == filters, "Expected %d filters but loaded kernel has shape %s for conv %s" % (kernel_size, kernel.shape, name)
            if isinstance(kernel_size, int):
                assert kernel.shape[0] == kernel.shape[1] == kernel_size
            else:
                assert kernel.shape[0:2] == kernel_size
            if isinstance(strides, int):
                strides = [1, strides, strides, 1]
            elif len(strides) == 2:
                strides = [1, strides[0], strides[1], 1]
            n = tf.nn.conv2d(n, kernel, strides=strides, padding=padding.upper(), name=name)
            if activation is not None:
                n = activation(n)
            n = tf.nn.bias_add(n, reader.get_tensor("%s/%s/bias" % (scope, name)))
            return n
    else:
        def conv(n, filters, kernel_size, strides=(1, 1), padding="valid", activation=None, name=None, kernel_initializer=None):
            with tf.variable_scope(scope):
                return tf.layers.conv2d(n, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                        activation=activation, name=name, reuse=tf.AUTO_REUSE, kernel_initializer=kernel_initializer)
    return conv
