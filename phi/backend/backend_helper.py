import six
from collections import namedtuple

import numpy as np

from .tensorop import expand, collapsed_gather_nd, CollapsedTensor as T, collapse


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


def general_grid_sample_nd(grid, coords, boundary, constant_values, math):
    """
    Backend-independent grid sampling with linear interpolation.
    :param grid: tensor of shape (batch_dim, spatial dims..., channels)
    :param coords: tensor of shape (batch_dim, ..., spatial_rank)
    :param boundary: 'zero'/'constant', 'replicate', 'circular', 'symmetric', 'reflect'
    :param constant_values: extrapolation values (same options as in pad)
    :param math: backend
    :return: resampled tensor
    """
    grid, coords, boundary = _pad_constant_boundaries(grid, coords, boundary, constant_values, math)

    resolution = np.array([int(d) for d in grid.shape[1:-1]])
    sp_rank = math.ndims(grid) - 2
    # --- Compute weights ---
    floor = math.floor(coords)
    up_weights = coords - floor
    lo_weights = math.unstack(1 - up_weights, axis=-1, keepdims=True)
    up_weights = math.unstack(up_weights, axis=-1, keepdims=True)
    lo_coords = math.cast(floor, np.int32)
    hi_coords = _apply_boundary(boundary, lo_coords + 1, resolution, math)
    lo_coords = _apply_boundary(boundary, lo_coords, resolution, math)

    def interpolate_nd(is_hi_by_axis, axis):
        is_hi_by_axis_2 = is_hi_by_axis | np.array([ax == axis for ax in range(sp_rank)])
        coords1 = math.where(is_hi_by_axis, hi_coords, lo_coords)
        coords2 = math.where(is_hi_by_axis_2, hi_coords, lo_coords)
        if axis == sp_rank - 1:
            lo_values = math.gather_nd(grid, coords1, batch_dims=1)
            up_values = math.gather_nd(grid, coords2, batch_dims=1)
        else:
            lo_values = interpolate_nd(is_hi_by_axis, axis + 1)
            up_values = interpolate_nd(is_hi_by_axis_2, axis + 1)
        return lo_values * lo_weights[axis] + up_values * up_weights[axis]
    result = interpolate_nd(np.array([False] * sp_rank), 0)
    return result


def _pad_constant_boundaries(grid, coords, boundary, constant_values, math):
    boundary = T(boundary)
    spatial_rank = math.staticshape(coords)[-1]
    pad_widths = [[1 if boundary[dim, upper] in ('zero', 'constant') else 0 for upper in (False, True)] for dim in range(-spatial_rank-1, -1)]
    boundary = [['replicate' if boundary[dim, upper] in ('zero', 'constant') else boundary[dim, upper] for upper in (False, True)] for dim in range(-spatial_rank-1, -1)]
    lower_pads = [lu[0] for lu in pad_widths]
    grid = math.pad(grid, [[0, 0]] + pad_widths + [[0, 0]], mode='constant', constant_values=constant_values)
    if sum(lower_pads) > 0:
        coords += lower_pads
    boundary = collapse(boundary)
    return grid, coords, boundary


def _circular(coords, input_size, math):
    return math.mod(math.mod(coords, input_size) + input_size, input_size)


def _apply_boundary(boundary, coords, input_size, math):
    if boundary == 'zero' or boundary == 'constant':
        raise ValueError("boundary 'zero' cannot be applied to coordinates")
    elif boundary == 'replicate':
        return math.maximum(math.minimum(coords, input_size - 1), 0)
    elif boundary == 'circular':
        return _circular(coords, input_size, math)
    elif boundary == 'symmetric':
        coords = _circular(coords, 2 * input_size, math)
        return ((2 * input_size - 1) - math.abs((2 * input_size - 1) - 2 * coords)) // 2
    elif boundary == 'reflect':
        coords = _circular(coords, 2 * input_size - 2, math)
        return (input_size - 1) - math.abs((input_size - 1) - coords)
    else:
        raise ValueError('Invalid boundary: %s' % boundary)
