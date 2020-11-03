from typing import Any

from phi import math
from phi.geom import Geometry, GridCell, Box
from ._field import SampledField
from ._grid import CenteredGrid
from ..geom._stack import GeometryStack
from ..math import Tensor


class PointCloud(SampledField):

    def __init__(self, elements: Geometry, values: Any = 1, extrapolation=math.extrapolation.ZERO, add_overlapping=False):
        """
        A point cloud consists of elements at arbitrary locations.
        A value or vector is associated with each element.

        Outside of elements, the value of the field is determined by the extrapolation.

        All points belonging to one example must be listed in the 'points' dimension.

        Unlike with GeometryMask, the elements of a PointCloud are assumed to be small.
        When sampling this field on a grid, scatter functions may be used.

        :param elements: Geometry object specifying the sample points and sizes
        :param values: values corresponding to elements
        :param extrapolation: values outside elements
        :param add_overlapping: True: values of overlapping geometries are summed. False: values between overlapping geometries are interpolated
        """
        SampledField.__init__(self, elements, values, extrapolation)
        self._add_overlapping = add_overlapping
        assert 'points' in self.shape, "Cannot create PointCloud without 'points' dimension. Add it either to elements or to values as batch dimension."

    def volume_sample(self, geometry: Geometry, reduce_channels=()) -> Tensor:
        if not reduce_channels:
            if geometry == self.elements:
                return self.values
            elif isinstance(geometry, GridCell):
                return self._grid_scatter(geometry.bounds, geometry.resolution)
            elif isinstance(geometry, GeometryStack):
                sampled = [self.sample_at(g) for g in geometry.geometries]
                return math.batch_stack(sampled, geometry.stack_dim_name)
            else:
                raise NotImplementedError()
        else:
            assert len(reduce_channels) == 1
            components = self.unstack('vector') if 'vector' in self.shape else (self,) * geometry.shape.get_size(reduce_channels[0])
            sampled = [c.volume_sample(p) for c, p in zip(components, geometry.unstack(reduce_channels[0]))]
            return math.channel_stack(sampled, 'vector')

    def sample_at(self, points, reduce_channels=()) -> Tensor:
        raise NotImplementedError()

    def _grid_scatter(self, box: Box, resolution: math.Shape):
        """
        Approximately samples this field on a regular grid using math.scatter().

        :param box: physical dimensions of the grid
        :param resolution: grid resolution
        :return: CenteredGrid
        """
        closest_index = math.to_int(math.round(box.global_to_local(self.points) * resolution))
        if self._add_overlapping:
            duplicates_handling = 'add'
        else:
            if self.values.shape.spatial_rank > 0:
                duplicates_handling = 'mean'
            else:
                duplicates_handling = 'any'  # constant value, no need for interpolation
        scattered = math.scatter(closest_index, self.values, resolution, duplicates_handling=duplicates_handling, outside_handling='discard', scatter_dims=('points',))
        return scattered

    def __repr__(self):
        return "PointCloud[%s]" % (self.shape,)


def _distribute_points(density, particles_per_cell=1, distribution='uniform'):
    """
Distribute points according to the distribution specified in density.
    :param density: binary tensor
    :param particles_per_cell: integer
    :param distribution: 'uniform' or 'center'
    :return: tensor of shape (batch_size, point_count, rank)
    """
    assert distribution in ('center', 'uniform')
    index_array = []
    batch_size = math.staticshape(density)[0] if math.staticshape(density)[0] is not None else 1
    
    for batch in range(batch_size):
        indices = math.where(density[batch, ..., 0] > 0)
        indices = math.to_float(indices)

        temp = []
        for _ in range(particles_per_cell):
            if distribution == 'center':
                temp.append(indices + 0.5)
            elif distribution == 'uniform':
                temp.append(indices + math.random_uniform(math.shape(indices)))
        index_array.append(math.concat(temp, axis=0))
    try:
        index_array = math.stack(index_array)
        return index_array
    except ValueError:
        raise ValueError("all arrays in the batch must have the same number of active cells.")
