import six
from collections import namedtuple

import numpy as np

from .tensorop import expand, collapsed_gather_nd, CollapsedTensor as CT, collapse


PadSettings = namedtuple('PadSettings', ['pad_width', 'mode', 'constant_values'])


def split_multi_mode_pad(tensor_rank, pad_settings, split_by_constant_value=True):
    dims = range(tensor_rank)
    pad_width, mode, constant_values = pad_settings
    if isinstance(mode, six.string_types):
        if not split_by_constant_value or not isinstance(constant_values, (tuple, list)):
            if isinstance(constant_values, (tuple, list)):
                constant_values = expand(constant_values, shape=(len(dims), 2))
            pad_width = expand(pad_width, [tensor_rank, 2])
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


NeighbourReduce = namedtuple('NeighbourReduce', ['requires_weights', 'f'])


def general_grid_sample_nd(grid, coords, boundary, constant_values, math, reduce='linear'):
    """
    Backend-independent grid sampling with linear interpolation.
    Supports boundary conditions per face: 'constant' , 'replicate', 'circular', 'symmetric', 'reflect'.

    Interpolation at the boundaries works according to the following principle:
    The boundary mode determines the value at virtual grid points outside the grid bounds.
    This is exact, even for far-away points.
    Then, linear interpolation is used to determine the point between grid points.
    Consequently, for constant boundaries, the value linearly approaches the constant value over the distance of one cell at the boundary.

    :param grid: tensor of shape (batch_dim, spatial dims..., channels)
    :param coords: tensor of shape (batch_dim, ..., spatial_rank)
    :param boundary: 'zero'/'constant', 'replicate', 'circular', 'symmetric', 'reflect'
    :param constant_values: extrapolation values (same options as in pad)
    :param math: backend
    :return: tensor of sampled values from the grid
    """
    if not isinstance(reduce, NeighbourReduce):
        reduce = {
            'linear': NeighbourReduce(True, lambda v1, v2, w1, w2: v1 * w1 + v2 * w2),
            'min': NeighbourReduce(False, lambda v1, v2: math.minimum(v1, v2)),
            'max': NeighbourReduce(False, lambda v1, v2: math.maximum(v1, v2)),
            'minmax': NeighbourReduce(False, lambda v1, v2: (math.minimum(v1[0], v2[0]), math.maximum(v1[1], v2[1])) if isinstance(v1, tuple) else (math.minimum(v1, v2), math.maximum(v1, v2))),
        }[reduce]
    grid, coords, boundary = pad_constant_boundaries(grid, coords, boundary, constant_values, math)

    resolution = np.array([int(d) for d in grid.shape[1:-1]])
    sp_rank = math.ndims(grid) - 2
    # --- Compute weights ---
    floor = math.floor(coords)
    lo_coords = math.to_int(floor)
    hi_coords = apply_boundary(boundary, lo_coords + 1, resolution, math)
    lo_coords = apply_boundary(boundary, lo_coords, resolution, math)
    if reduce.requires_weights:
        hi_weights = coords - floor
        lo_weights = math.unstack(1 - hi_weights, axis=-1, keepdims=True)
        hi_weights = math.unstack(hi_weights, axis=-1, keepdims=True)

    def interpolate_nd(is_hi_by_axis, axis):
        is_hi_by_axis_2 = is_hi_by_axis | np.array([ax == axis for ax in range(sp_rank)])
        coords1 = math.where(is_hi_by_axis, hi_coords, lo_coords)
        coords2 = math.where(is_hi_by_axis_2, hi_coords, lo_coords)
        if axis == sp_rank - 1:
            lo_values = math.gather_nd(grid, coords1, batch_dims=1)
            hi_values = math.gather_nd(grid, coords2, batch_dims=1)
        else:
            lo_values = interpolate_nd(is_hi_by_axis, axis + 1)
            hi_values = interpolate_nd(is_hi_by_axis_2, axis + 1)
        if reduce.requires_weights:
            return reduce.f(lo_values, hi_values, lo_weights[axis], hi_weights[axis])
        else:
            return reduce.f(lo_values, hi_values)
    result = interpolate_nd(np.array([False] * sp_rank), 0)
    return result


def pad_constant_boundaries(grid, coords, boundary, constant_values, math):
    boundary = CT(boundary)
    spatial_rank = math.staticshape(coords)[-1]
    pad_widths = [[1 if boundary[dim, upper] in ('zero', 'constant') else 0 for upper in (False, True)] for dim in range(-spatial_rank - 1, -1)]
    boundary = [['replicate' if boundary[dim, upper] in ('zero', 'constant') else boundary[dim, upper] for upper in (False, True)] for dim in range(-spatial_rank - 1, -1)]
    lower_pads = [lu[0] for lu in pad_widths]
    grid = math.pad(grid, [[0, 0]] + pad_widths + [[0, 0]], mode='constant', constant_values=constant_values)
    if sum(lower_pads) > 0:
        coords = math.add(coords, math.cast(lower_pads, math.dtype(coords)))
    boundary = collapse(boundary)
    return grid, coords, boundary


def apply_boundary(boundary, coords, input_size, math):
    if isinstance(boundary, six.string_types):
        return _apply_single_boundary(boundary, coords, input_size, math)
    coords = math.unstack(coords, axis=-1)
    assert len(boundary) == len(input_size) == len(coords)
    boundary = CT(boundary)
    result = []
    for dim, dim_coords in enumerate(coords):
        if boundary[dim, 0] == boundary[dim, 1]:
            result.append(_apply_single_boundary(boundary[dim, 0], dim_coords, input_size[dim], math))
        else:  # separate boundary for lower and upper face
            lower = _apply_single_boundary(boundary[dim, 0], dim_coords, input_size[dim], math)
            upper = _apply_single_boundary(boundary[dim, 1], dim_coords, input_size[dim], math)
            result.append(math.where(dim_coords <= 0, lower, upper))
    return math.stack(result, axis=-1)


def _apply_single_boundary(boundary, coords, input_size, math):
    if boundary == 'zero' or boundary == 'constant':
        raise ValueError("boundary 'zero' cannot be applied to coordinates")
    elif boundary == 'replicate':
        return math.clip(coords, 0, input_size - 1)
    elif boundary == 'circular':
        return math.mod(coords, input_size)
    elif boundary == 'symmetric':
        coords = math.mod(coords, 2 * input_size)
        return ((2 * input_size - 1) - math.abs((2 * input_size - 1) - 2 * coords)) // 2
    elif boundary == 'reflect':
        coords = math.mod(coords, 2 * input_size - 2)
        return (input_size - 1) - math.abs((input_size - 1) - coords)
    else:
        raise ValueError('Invalid boundary: %s' % (boundary,))


def equalize_ranks(tensors, math):
    rank = max([math.ndims(tensor) for tensor in tensors])
    return [math.expand_dims(tensor, 0, number=rank - math.ndims(tensor)) for tensor in tensors]


def equalize_shapes(tensors, math, ignore_outer_dims=0):
    tensors = equalize_ranks(tensors, math)
    shape = _combined_shape([math.staticshape(t) for t in tensors], ignore_outer_dims)
    result = []
    for tensor in tensors:
        tensor_shape = math.staticshape(tensor)
        multiples = [1] * ignore_outer_dims
        for dim in range(ignore_outer_dims, len(tensor_shape)):
            multiples.append(shape[dim] // tensor_shape[dim])
        result.append(math.tile(tensor, multiples))
    return result


def _combined_shape(shapes, ignore_outer_dims):
    rank = len(shapes[0])  # assume all shapes have same rank
    resulting_shape = [None] * ignore_outer_dims
    for dim in range(ignore_outer_dims, rank):
        dim_value = None
        for shape in shapes:
            dim_value = combined_dim(dim_value, shape[dim])
        resulting_shape.append(dim_value)
    return tuple(resulting_shape)


def combined_dim(dim1, dim2):
    if dim1 is None and dim2 is None:
        return None
    if dim1 is None or dim1 == 1:
        return dim2
    if dim2 is None or dim2 == 1:
        return dim1
    assert dim1 == dim2, "Cannot bring shapes together because dimensions are incompatible: %d and %d" % (dim1, dim2)
    return dim1


def circular_pad(value, pad_width, math):
    dims = range(math.ndims(value))
    for dim in dims:
        pad_lower, pad_upper = pad_width[dim]
        if pad_lower == 0 and pad_upper == 0:
            continue  # Nothing to pad
        lower = value[tuple([slice(value.shape[dim] - pad_lower, None) if d == dim else slice(None) for d in dims])]
        upper = value[tuple([slice(None, pad_upper) if d == dim else slice(None) for d in dims])]
        value = math.concat([lower, value, upper], axis=dim)
    return value


def replicate_pad(value, pad_width, math):
    dims = range(math.ndims(value))
    for dim in dims:
        pad_lower, pad_upper = pad_width[dim]
        if pad_lower == 0 and pad_upper == 0:
            continue  # Nothing to pad
        bottom_row = value[(slice(None),) + tuple([slice(1) if d == dim else slice(None) for d in dims]) + (slice(None),)]
        top_row = value[(slice(None),) + tuple([slice(-1, None) if d == dim else slice(None) for d in dims]) + (slice(None),)]
        value = math.concat([bottom_row] * pad_lower + [value] + [top_row] * pad_upper)
    return value


def symmetric_pad(value, pad_width, math):
    raise NotImplementedError()  # only used by PyTorch which does not support ::-1 axis flips
    dims = range(math.ndims(value))
    for dim in dims:
        pad_lower, pad_upper = pad_width[dim]
        if pad_lower == 0 and pad_upper == 0:
            continue  # Nothing to pad
        top_rows = value[tuple([slice(value.shape[dim] - pad_upper, None) if d == dim else slice(None) for d in dims])]
        bottom_rows = value[tuple([slice(None, pad_lower) if d == dim else slice(None) for d in dims])]
        top_rows = math.flip_axis(top_rows, dim)
        bottom_rows = math.flip_axis(bottom_rows, dim)
        value = math.concat([bottom_rows, value, top_rows], axis=dim)
    return value
