from typing import Any

from phi import math
from phi.geom import Geometry, GridCell, Box
from ._field import SampledField
from ..geom._stack import GeometryStack
from ..math import Tensor, instance


class PointCloud(SampledField):
    """
    A point cloud consists of elements at arbitrary locations.
    A value or vector is associated with each element.

    Outside of elements, the value of the field is determined by the extrapolation.

    All points belonging to one example must be listed in the 'points' dimension.

    Unlike with GeometryMask, the elements of a PointCloud are assumed to be small.
    When sampling this field on a grid, scatter functions may be used.

    See the `phi.field` module documentation at https://tum-pbs.github.io/PhiFlow/Fields.html
    """

    def __init__(self,
                 elements: Geometry,
                 values: Any = 1,
                 extrapolation=math.extrapolation.ZERO,
                 add_overlapping=False,
                 bounds: Box = None,
                 color: str or Tensor or tuple or list or None = None):
        """
        Args:
          elements: Geometry object specifying the sample points and sizes
          values: values corresponding to elements
          extrapolation: values outside elements
          add_overlapping: True: values of overlapping geometries are summed. False: values between overlapping geometries are interpolated
          bounds: (optional) size of the fixed domain in which the points should get visualized. None results in max and min coordinates of points.
          color: (optional) hex code for color or tensor of colors (same length as elements) in which points should get plotted.
        """
        SampledField.__init__(self, elements, math.wrap(values), extrapolation)
        self._add_overlapping = add_overlapping
        assert bounds is None or isinstance(bounds, Box), 'Invalid bounds.'
        self._bounds = bounds
        color = '#0060ff' if color is None else color
        self._color = math.wrap(color, instance('points')) if isinstance(color, (tuple, list)) else math.wrap(color)

    @property
    def shape(self):
        return self._elements.shape & self._values.shape.non_spatial

    def __getitem__(self, item: dict):
        elements = self.elements[item]
        values = self._values[item]
        color = self._color[item]
        extrapolation = self._extrapolation[item]
        return PointCloud(elements, values, extrapolation, self._add_overlapping, self._bounds, color)

    def with_elements(self, elements: Geometry):
        return PointCloud(elements=elements, values=self.values, extrapolation=self.extrapolation, add_overlapping=self._add_overlapping, bounds=self._bounds, color=self._color)

    def with_values(self, values):
        return PointCloud(elements=self.elements, values=values, extrapolation=self.extrapolation, add_overlapping=self._add_overlapping, bounds=self._bounds, color=self._color)

    def with_extrapolation(self, extrapolation: math.Extrapolation):
        return PointCloud(elements=self.elements, values=self.values, extrapolation=extrapolation, add_overlapping=self._add_overlapping, bounds=self._bounds, color=self._color)

    def with_color(self, color: str or Tensor or tuple or list):
        return PointCloud(elements=self.elements, values=self.values, extrapolation=self.extrapolation, add_overlapping=self._add_overlapping, bounds=self._bounds, color=color)

    def with_bounds(self, bounds: Box):
        return PointCloud(elements=self.elements, values=self.values, extrapolation=self.extrapolation, add_overlapping=self._add_overlapping, bounds=bounds, color=self._color)

    def __value_attrs__(self):
        return '_values', '_extrapolation'

    def __variable_attrs__(self):
        return '_values', '_elements'

    @property
    def bounds(self) -> Box:
        return self._bounds

    @property
    def color(self) -> Tensor:
        return self._color

    def _sample(self, geometry: Geometry) -> Tensor:
        if geometry == self.elements:
            return self.values
        elif isinstance(geometry, GridCell):
            return self._grid_scatter(geometry.bounds, geometry.resolution)
        elif isinstance(geometry, GeometryStack):
            sampled = [self._sample(g) for g in geometry.geometries]
            return math.stack(sampled, geometry.stack_dim)
        else:
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
        closest_index = box.global_to_local(self.points) * resolution - 0.5
        mode = 'add' if self._add_overlapping else 'mean'
        base = math.zeros(resolution)
        if isinstance(self.extrapolation, math.extrapolation.ConstantExtrapolation):
            base += self.extrapolation.value
        scattered = math.scatter(base, closest_index, self.values, mode=mode, outside_handling='discard')
        return scattered

    def __repr__(self):
        return "PointCloud[%s]" % (self.shape,)

    def __and__(self, other):
        assert isinstance(other, PointCloud)
        from ._field_math import concat
        return concat([self, other], instance('points'))


def nonzero(field: SampledField):
    indices = math.nonzero(field.values, list_dim=instance('points'))
    elements = field.elements[indices]
    return PointCloud(elements, values=math.tensor(1.), extrapolation=math.extrapolation.ZERO, add_overlapping=False, bounds=field.bounds, color=None)
