from typing import Any

from phi import math
from phi.geom import Geometry, GridCell, Box
from ._field import SampledField
from ..geom._stack import GeometryStack
from ..math import Tensor


class PointCloud(SampledField):

    def __init__(self, elements: Geometry,
                 values: Any = 1,
                 extrapolation=math.extrapolation.ZERO,
                 add_overlapping=False,
                 bounds: Box = None,
                 color: str or Tensor or tuple or list or None = None):
        """
        A point cloud consists of elements at arbitrary locations.
        A value or vector is associated with each element.

        Outside of elements, the value of the field is determined by the extrapolation.

        All points belonging to one example must be listed in the 'points' dimension.

        Unlike with GeometryMask, the elements of a PointCloud are assumed to be small.
        When sampling this field on a grid, scatter functions may be used.

        See the `phi.field` module documentation at https://tum-pbs.github.io/PhiFlow/Fields.html

        Args:
          elements: Geometry object specifying the sample points and sizes
          values: values corresponding to elements
          extrapolation: values outside elements
          add_overlapping: True: values of overlapping geometries are summed. False: values between overlapping geometries are interpolated
          bounds: (optional) size of the fixed domain in which the points should get visualized. None results in max and min coordinates of points.
          color: (optional) hex code for color or tensor of colors (same length as elements) in which points should get plotted.
        """
        SampledField.__init__(self, elements, values, extrapolation)
        self._add_overlapping = add_overlapping
        assert bounds is None or isinstance(bounds, Box), 'Invalid bounds.'
        self._bounds = bounds
        assert 'points' in self.shape, "Cannot create PointCloud without 'points' dimension. Add it either to elements or to values as batch dimension."
        if color is None:
            color = '#0060ff'
        self._color = math.tensor(color, names='points') if isinstance(color, (tuple, list)) else math.tensor(color)

    @property
    def bounds(self) -> Box:
        return self._bounds

    @property
    def color(self) -> Tensor:
        return self._color

    def sample_in(self, geometry: Geometry, reduce_channels=()) -> Tensor:
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
            sampled = [c.sample_in(p) for c, p in zip(components, geometry.unstack(reduce_channels[0]))]
            return math.channel_stack(sampled, 'vector')

    def sample_at(self, points, reduce_channels=()) -> Tensor:
        raise NotImplementedError()

    def _grid_scatter(self, box: Box, resolution: math.Shape):
        """
        Approximately samples this field on a regular grid using math.scatter().

        Args:
          box: physical dimensions of the grid
          resolution: grid resolution
          box: Box: 
          resolution: math.Shape: 

        Returns:
          CenteredGrid

        """
        closest_index = math.to_int(math.round(box.global_to_local(self.points) * resolution - 0.5))
        if self._add_overlapping:
            duplicates_handling = 'add'
        else:
            duplicates_handling = 'mean'
        scattered = math.scatter(closest_index, self.values, resolution, duplicates_handling=duplicates_handling, outside_handling='discard', scatter_dims=('points',))
        return scattered

    def __repr__(self):
        return "PointCloud[%s]" % (self.shape,)

    def __and__(self, other):
        assert isinstance(other, PointCloud)
        from ._field_math import concat
        return concat(self, other, dim='points')
