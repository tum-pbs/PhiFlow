import warnings

import numpy as np
import tensorflow as tf

from phi import struct, math
from phi.math._tensors import NativeTensor
from phi.field import StaggeredGrid, CenteredGrid
from . import TF_BACKEND


def GradientTape(watch=(), persistent=False) -> tf.GradientTape:
    tape = tf.GradientTape(persistent)

    with tape:
        for value in watch:
            assert isinstance(value, math.Tensor), value
            value._op1(lambda native: tape.watch(native))
    return tape


def gradients(target: math.Tensor, sources: math.Tensor, gradient_tape: tf.GradientTape = None, output_gradients=None):
    assert isinstance(target, NativeTensor)
    target = target.native()
    if gradient_tape is None:
        raise NotImplementedError()
    if output_gradients is not None:
        raise NotImplementedError()
    sources_list = []
    sources._op1(lambda native: sources_list.append(native))
    grads = list(gradient_tape.gradient(target, sources_list))
    for i, grad in enumerate(grads):
        assert grad is not None, f"Missing gradient for source with shape {sources_list[i].shape}"
    grads = sources._op1(lambda native: grads.pop(0))
    return grads


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
        return istensor(obj.values)
    if isinstance(obj, StaggeredGrid):
        return np.any([istensor(t) for t in obj.values])
    return isinstance(obj, (tf.Tensor, tf.Variable))
