from .staggered_grid import StaggeredGrid, unstack_staggered_tensor
from .grid import CenteredGrid
from phi import math
from phi import geom


def staggered_grid(tensor, name=None):
    tensor = tensor[...,::-1]  # manta: xyz, phiflow: zyx
    assert math.staticshape(tensor)[-1] == math.spatial_rank(tensor)
    resolution = [d-1 for d in math.staticshape(tensor)[1:-1]]
    tensors = unstack_staggered_tensor(tensor)
    return StaggeredGrid.from_tensors(name, geom.AABox(0, resolution), tensors)


def centered_grid(tensor, name=None, crop_valid=False):
    if crop_valid:
        tensor = tensor[(slice(None),) + (slice(-1),)*math.spatial_rank(tensor) + (slice(None),)]
    resolution = math.staticshape(tensor)[1:-1]
    return CenteredGrid(name, geom.AABox(0, resolution), tensor)
