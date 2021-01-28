from collections import namedtuple

from .tensorop import CollapsedTensor as CT, collapse


NeighbourReduce = namedtuple('NeighbourReduce', ['requires_weights', 'f'])


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


def combined_dim(dim1, dim2, type_str: str = 'batch'):
    if dim1 is None and dim2 is None:
        return None
    if dim1 is None or dim1 == 1:
        return dim2
    if dim2 is None or dim2 == 1:
        return dim1
    assert dim1 == dim2, f"Incompatible {type_str} dimensions: x0 {dim1}, y {dim2}"
    return dim1
