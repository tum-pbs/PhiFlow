import six
from collections import namedtuple

import numpy as np

from .tensorop import CollapsedTensor as CT, collapse


NeighbourReduce = namedtuple('NeighbourReduce', ['requires_weights', 'f'])


def general_grid_sample_nd(grid, coords, boundary, constant_values, math, reduce='linear'):
    """
    Backend-independent grid sampling with linear interpolation.
    Supports boundary conditions per face: 'constant' , 'boundary', 'periodic', 'symmetric', 'reflect'.

    Interpolation at the boundaries works according to the following principle:
    The boundary mode determines the value at virtual grid points outside the grid bounds.
    This is exact, even for far-away points.
    Then, linear interpolation is used to determine the point between grid points.
    Consequently, for constant boundaries, the value linearly approaches the constant value over the distance of one cell at the boundary.

    :param grid: tensor of shape (batch_dim, spatial dims..., channels)
    :param coords: tensor of shape (batch_dim, ..., spatial_rank)
    :param boundary: 'zero'/'constant', 'boundary', 'periodic', 'symmetric', 'reflect'
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
    pad_widths = [[1 if boundary[dim, upper] == 'constant' else 0 for upper in (False, True)] for dim in range(-spatial_rank - 1, -1)]
    boundary = [['boundary' if boundary[dim, upper] == 'constant' else boundary[dim, upper] for upper in (False, True)] for dim in range(-spatial_rank - 1, -1)]
    lower_pads = [lu[0] for lu in pad_widths]
    grid = math.pad(grid, [[0, 0]] + pad_widths + [[0, 0]], mode='constant', constant_values=constant_values)
    if sum(lower_pads) > 0:
        coords = math.add(coords, math.cast(lower_pads, math.dtype(coords)))
    boundary = collapse(boundary)
    return grid, coords, boundary


def combined_dim(dim1, dim2):
    if dim1 is None and dim2 is None:
        return None
    if dim1 is None or dim1 == 1:
        return dim2
    if dim2 is None or dim2 == 1:
        return dim1
    assert dim1 == dim2, "Cannot bring shapes together because dimensions are incompatible: %d and %d" % (dim1, dim2)
    return dim1
