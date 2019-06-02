# coding=utf-8
import tensorflow as tf
from phi.math.nd import *


def group_normalization(x, group_count, eps=1e-5):
    batch_size, H, W, C = tf.shape(x)
    gamma = tf.Variable(np.ones([1,1,1,C]), dtype=tf.float32, name="GN_gamma")
    beta = tf.Variable(np.zeros([1,1,1,C]), dtype=tf.float32, name="GN_beta")
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
    y = tf.pad(y, [[0,0], pad1, pad2, [0,0]], mode=padding)
    y = tf.layers.conv2d(y, nb_channels, kernel_size=kernel_size, strides=_strides, padding='valid',
             name=None if name is None else name+"/conv1", trainable=trainable, reuse=reuse)
    # y = tf.layers.batch_normalization(y, name=None if name is None else name+"/norm1", training=training, trainable=trainable, reuse=reuse)
    y = activation(y)

    y = tf.pad(y, [[0,0], pad1, pad2, [0,0]], mode=padding)
    y = tf.layers.conv2d(y, nb_channels, kernel_size=kernel_size, strides=(1, 1), padding='valid',
             name=None if name is None else name + "/conv2", trainable=trainable, reuse=reuse)
    # y = tf.layers.batch_normalization(y, name=None if name is None else name+"/norm2", training=training, trainable=trainable, reuse=reuse)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = tf.pad(shortcut, [[0,0], pad1, pad2, [0,0]], mode=padding)
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
    y = tf.pad(y, [[0,0], pad1, [0,0]], mode=padding)
    y = tf.layers.conv1d(y, nb_channels, kernel_size=kernel_size, strides=_strides, padding='valid',
             name=None if name is None else name+"/conv1", trainable=trainable, reuse=reuse)
    # y = tf.layers.batch_normalization(y, name=None if name is None else name+"/norm1", training=training, trainable=trainable, reuse=reuse)
    y = activation(y)

    y = tf.pad(y, [[0,0], pad1, [0,0]], mode=padding)
    y = tf.layers.conv1d(y, nb_channels, kernel_size=kernel_size, strides=(1,), padding='valid',
             name=None if name is None else name + "/conv2", trainable=trainable, reuse=reuse)
    # y = tf.layers.batch_normalization(y, name=None if name is None else name+"/norm2", training=training, trainable=trainable, reuse=reuse)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1,):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = tf.pad(shortcut, [[0,0], pad1, [0,0]], mode=padding)
        shortcut = tf.layers.conv1d(shortcut, nb_channels, kernel_size=(1, 1), strides=_strides, padding='valid',
                        name=None if name is None else name + "/convid", trainable=trainable, reuse=reuse)
        # shortcut = tf.layers.batch_normalization(shortcut, name=None if name is None else name+"/normid", training=training, trainable=trainable, reuse=reuse)

    y += shortcut
    y = activation(y)

    return y


def istensor(object):
    if isinstance(object, StaggeredGrid):
        object = object.staggered
    return isinstance(object, tf.Tensor)
