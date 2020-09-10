import numpy as np
from phi import math
from . import tf
from ..physics.field import Field, CenteredGrid, StaggeredGrid
from ..physics.domain import Domain
from .grid_layers import conv_layer, residual_block


def u_net(domain, input_fields, output_field, levels=2, filters=16, blocks_per_level=2, skip_combine='concat', training=False, trainable=True, reuse=None):
    """
Restrictions:
- 2D only
- Domain resolution must be multiple of 2**levels

    :param skip_combine: 'concat'
    :param blocks_per_level: number of residual blocks per level
    :param filters: Number of convolutional filters
    :type filters: int or tuple or list
    :param levels: number of additional resolution levels, equals number of downsampling / upsampling operations
    :param domain: the u-net is executed on this domain.
    :type domain: Domain
    :param input_fields: list of Fields to be passed to the network as input
    :param output_field: determines sample points of the result
    :param training: whether the network is executed in training or inference mode
    :param trainable: whether the weights of the network are trainable
    :param reuse: whether to reuse weights from previous unet calls
    :return: Field sampled like output_field
    """
    assert isinstance(domain, Domain)
    assert isinstance(output_field, Field)
    net_inputs = []
    for input_field in input_fields:
        assert isinstance(input_field, Field)
        resampled = input_field.at(domain)
        net_inputs.append(resampled)
    y = CenteredGrid.sample(math.concat([math.to_float(grid.data) for grid in net_inputs], axis=-1), domain)
    # --- Execute network ---
    pad_width = sum([2 ** i for i in range(levels)])
    y = y.padded([[0, pad_width]] * domain.rank)
    resolutions = [y]
    for level in range(levels):
        level_filters = filters if isinstance(filters, int) else filters[level]
        y = conv_layer(resolutions[0], level_filters, 2, strides=2, activation=tf.nn.relu, padding='valid', name='down_convolution_%d' % level, trainable=trainable, reuse=reuse)
        for i in range(blocks_per_level):
            y = residual_block(y, level_filters, name='down_res_block_%d_%d' % (level, i), training=training, trainable=trainable, reuse=reuse)
        resolutions.insert(0, y)

    y = resolutions.pop(0)
    assert np.all(y.box.size == domain.box.size)

    for level in range(levels):
        y = math.upsample2x(y)
        res_in = resolutions.pop(0)
        res_in = res_in.at(y)  # No resampling required, simply shaving off the top rows
        if skip_combine == 'concat':
            y = y.with_data(math.concat([y.data, res_in.data], axis=-1))
        else:
            raise NotImplementedError()
            y = y + res_in
        y = y.padded([[0, 1]] * y.rank)
        if resolutions:
            level_filters = filters if isinstance(filters, int) else reversed(filters)[level]
            y = conv_layer(y, level_filters, kernel_size=2, activation=tf.nn.relu, padding='valid', name='up_convolution_%d' % level, trainable=trainable, reuse=reuse)
            for i in range(blocks_per_level):
                y = residual_block(y, level_filters, name='up_res_block_%d_%d' % (level, i), training=training, trainable=trainable, reuse=reuse)
        else:  # Last iteration
            y = conv_layer(y, output_field.component_count, kernel_size=2, activation=None, padding='valid', name='up_convolution_%d' % level, trainable=trainable, reuse=reuse)
    result = y.at(output_field)
    return result
