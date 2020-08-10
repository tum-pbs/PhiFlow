from .staggered_grid import StaggeredGrid, unstack_staggered_tensor
from .grid import CenteredGrid
from phi import math
from phi import geom


def staggered_grid(tensor, name='manta_staggered'):
    tensor = tensor[...,::-1]  # manta: xyz, phiflow: zyx
    assert math.staticshape(tensor)[-1] == math.spatial_rank(tensor)
    return StaggeredGrid(tensor, name=name)


def centered_grid(tensor, name='manta_centered', crop_valid=False):
    if crop_valid:
        tensor = tensor[(slice(None),) + (slice(-1),) * math.spatial_rank(tensor) + (slice(None),)]
    return CenteredGrid(tensor, name=name)
