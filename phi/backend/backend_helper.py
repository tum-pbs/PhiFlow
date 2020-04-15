import six
from collections import namedtuple

import numpy as np

from .tensorop import expand, collapsed_gather_nd


PadSettings = namedtuple('PadSettings', ['pad_width', 'mode', 'constant_values'])


def split_multi_mode_pad(tensor_rank, pad_settings, split_by_constant_value=True):
    dims = range(tensor_rank)
    pad_width, mode, constant_values = pad_settings
    if isinstance(mode, six.string_types):
        if not split_by_constant_value or not isinstance(constant_values, (tuple, list)):
            if isinstance(constant_values, (tuple, list)):
                constant_values = expand(constant_values, shape=(len(dims), 2))
            return [PadSettings(pad_width, mode, constant_values)]
    mode = expand(mode, shape=(len(dims), 2))
    constant_values = expand(constant_values, shape=(len(dims), 2))
    passes = [('circular', 0), ('wrap', 0), ('replicate', 0), ('symmetric', 0), ('reflect', 0)]
    if split_by_constant_value:
        constant_value_set = set()
        for dim in dims:
            for upper in (False, True):
                constant_value_set.add(constant_values[dim][upper])
        for const in constant_value_set:
            passes.append(('constant', const))
    else:
        passes.append(('constant', constant_values))
    result = []  # list of PadSettings
    for single_mode, constant_value in passes:  # order matters! circular/wrap first
        widths = [[collapsed_gather_nd(pad_width, [dim, upper]) if mode[dim][upper] == single_mode and constant_values[dim][upper] == collapsed_gather_nd(constant_value, [dim, upper]) else 0 for upper in (False, True)] for dim in dims]
        if np.sum(np.array(widths)) > 0:
            result.append(PadSettings(widths, single_mode, constant_value))
    if np.sum(np.array(pad_width)) > 0 and len(result) == 0:
        split_multi_mode_pad(tensor_rank, pad_settings, split_by_constant_value=split_by_constant_value)

    return result
