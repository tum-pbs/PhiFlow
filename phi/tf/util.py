import warnings

import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

from phi import struct, math
from phi.field import StaggeredGrid, CenteredGrid
from . import TF_BACKEND


def _tf_name(trace, basename):
    path = trace.path('/')
    if basename is None and len(path) == 0:
        return None
    result = path if basename is None else basename + '/' + path
    return result


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
