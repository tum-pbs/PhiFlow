# coding=utf-8
import logging
import warnings
import numpy as np
from tensorflow.python import pywrap_tensorflow
from . import tf, TF_BACKEND

from phi import struct, math
from phi.math.math_util import is_static_shape
from phi.physics.field.staggered_grid import StaggeredGrid
from phi.physics.field.grid import CenteredGrid


def _tf_name(trace, basename):
    path = trace.path('/')
    if basename is None and len(path) == 0:
        return None
    result = path if basename is None else basename + '/' + path
    return result


def placeholder(shape, dtype=None, basename='Placeholder'):
    if struct.isstruct(dtype):
        def placeholder_map(trace):
            shape, dtype = trace.value
            return tf.placeholder(dtype, shape, _tf_name(trace, basename))
        zipped = struct.zip([shape, dtype], leaf_condition=is_static_shape)
        return struct.map(placeholder_map, zipped, leaf_condition=is_static_shape, trace=True)
    else:
        def f(trace): return tf.placeholder(TF_BACKEND.precision_dtype if dtype is None else dtype, trace.value, _tf_name(trace, basename))
        return struct.map(f, shape, leaf_condition=is_static_shape, trace=True)


def placeholder_like(obj, basename='Placeholder'):
    warnings.warn("placeholder_like may not respect the batch dimension. "
                  "For State objects, use placeholder(state.shape) instead.", DeprecationWarning, stacklevel=2)

    def f(attr): return tf.placeholder(attr.value.dtype, attr.value.shape, _tf_name(attr, basename))
    return struct.map(f, obj, leaf_condition=is_static_shape, trace=True)


def variable(initial_value, dtype=None, basename='Variable', trainable=True):
    def f(attr): return tf.Variable(attr.value, name=_tf_name(attr, basename), dtype=TF_BACKEND.precision_dtype if dtype is None else dtype, trainable=trainable)
    return struct.map(f, initial_value, trace=True)


def variable_generator(initializer, dtype=None, basename='Variable', trainable=True):
    def create_variable(shape):
        initial_value = initializer(shape)
        return variable(initial_value, dtype, basename, trainable)
    return create_variable


def constant(value, dtype=None, basename='const'):
    def f(trace): return tf.constant(trace.value, dtype=TF_BACKEND.precision_dtype if dtype is None else dtype, name=_tf_name(trace, basename))
    return struct.map(f, value, trace=True)


def is_placeholder(obj):
    return isinstance(obj, tf.Tensor) and obj.op.type == 'Placeholder'


isplaceholder = is_placeholder


def dataset_handle(shape, dtype, frames=None):
    """
Creates a single virtual TensorFlow dataset (iterator_handle) for the given struct.
The dataset is expected to hold contain all fields required for loading the obj given the current context item condition.
From the dataset, graph input tensors are derived and arranged into a struct of the same shape as obj.
If an integer is passed to frames, a list of such structs is created by unstacking the second-outer-most dimension of the dataset.
    :param shape: tensor shape or struct of tensor shapes
    :param dtype: data type of struct of data types matching shape
    :param frames: Number of frames contained in each example of the dataset. Expects shape (batch_size, frames, ...)
    :type frames: int or None
    :return: list of struct and placeholder.
     1. If frames=None: valid struct corresponding to obj. If frames>1: list thereof
     2. placeholder for a TensorFlow dataset iterator handle (dtype=string)
    :rtype: tuple
    """
    shapes = tuple(struct.flatten(shape, leaf_condition=is_static_shape))
    if struct.isstruct(dtype):
        dtypes = tuple(struct.flatten(dtype))
        assert len(dtypes) == len(shapes)
    else:
        dtypes = [dtype] * len(shapes)
    if frames is not None:
        shapes = tuple([shape[0:1] + (frames,) + shape[1:] for shape in shapes])
    # --- TF Dataset handle from string ---
    iterator_handle = tf.placeholder(tf.string, shape=[], name='dataset_iterator_handle')
    iterator = tf.data.Iterator.from_string_handle(iterator_handle, output_types=dtypes, output_shapes=shapes)
    next_element = iterator.get_next()
    # --- Create resulting struct by splitting `next_element`s ---
    if frames is None:
        next_element_list = list(next_element)
        next_struct = struct.map(lambda _: next_element_list.pop(0), shape, leaf_condition=is_static_shape)
    else:
        # --- Remap structures -> to `frames` long list of structs ---
        next_struct = []
        for frame_idx in range(frames):
            next_element_list = list(next_element)
            frame_struct = struct.map(lambda _: next_element_list.pop(0)[:, frame_idx, ...], shape, leaf_condition=is_static_shape)
            next_struct.append(frame_struct)
    return next_struct, iterator_handle


def group_normalization(x, group_count, eps=1e-5):
    batch_size, H, W, C = tf.shape(x)
    gamma = tf.Variable(np.ones([1, 1, 1, C]), dtype=TF_BACKEND.precision_dtype, name="GN_gamma")
    beta = tf.Variable(np.zeros([1, 1, 1, C]), dtype=TF_BACKEND.precision_dtype, name="GN_beta")
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
    warnings.warn("istensor is deprecated, use phi.tf.app.is_tensorflow_field instead", DeprecationWarning)
    if isinstance(obj, CenteredGrid):
        return istensor(obj.data)
    if isinstance(obj, StaggeredGrid):
        return np.any([istensor(t) for t in obj.data])
    return isinstance(obj, (tf.Tensor, tf.Variable))


def gradients(y, xs, grad_y=None):
    """
    Compute the analytic gradients using TensorFlow's automatic differentiation.

    :param y: tensor or struct of tensors. The contributions of all tensors in `y` are added up.
    :param xs: struct of input tensors
    :return: struct compatible with `xs` holding dy/dx
    """
    ys = struct.flatten(y)
    if grad_y is not None:
        grad_y = struct.flatten(grad_y)
        for i in range(len(grad_y)):
            grad_y[i] = math.cast(grad_y[i], math.dtype(ys[i]))
    xs_ = struct.flatten(xs)
    grad = tf.gradients(ys, xs_, grad_ys=grad_y)
    return struct.unflatten(grad, xs)


def stop_gradient(x):
    return struct.map(tf.stop_gradient, x)


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
