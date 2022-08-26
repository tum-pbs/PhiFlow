import warnings
from typing import Any

from phi import math
from phi.geom import Geometry, GridCell, Box, Point
from ._field import SampledField
from .numerical import Scheme
from ..geom._stack import GeometryStack
from ..math import Tensor, instance, Shape
from ..math.extrapolation import Extrapolation
from ..math.magic import slicing_dict


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
                 elements: Tensor or Geometry,
                 values: Any = 1.,
                 extrapolation: float or Extrapolation = 0.,
                 add_overlapping=False,
                 bounds: Box = None,
                 color: str or Tensor or tuple or list or None = None):
        """
        Args:
          elements: `Tensor` or `Geometry` object specifying the sample points and sizes
          values: values corresponding to elements
          extrapolation: values outside elements
          add_overlapping: True: values of overlapping geometries are summed. False: values between overlapping geometries are interpolated
          bounds: (optional) size of the fixed domain in which the points should get visualized. None results in max and min coordinates of points.
          color: (optional) hex code for color or tensor of colors (same length as elements) in which points should get plotted.
        """
        SampledField.__init__(self, elements, math.wrap(values), extrapolation, bounds)
        self._add_overlapping = add_overlapping
        color = '#0060ff' if color is None else color
        self._color = math.wrap(color, instance('points')) if isinstance(color, (tuple, list)) else math.wrap(color)

    @property
    def shape(self):
        return self._elements.shape.without('vector') & self._values.shape

    def __getitem__(self, item):
        item = slicing_dict(self, item)
        if not item:
            return self
        elements = self.elements[{dim: selection for dim, selection in item.items() if dim != 'vector'}]
        values = self._values[item]
        color = self._color[item]
        extrapolation = self._extrapolation[item]
        return PointCloud(elements, values, extrapolation, self._add_overlapping, self._bounds, color)

    def with_elements(self, elements: Geometry):
        return PointCloud(elements=elements, values=self.values, extrapolation=self.extrapolation, add_overlapping=self._add_overlapping, bounds=self._bounds, color=self._color)

    def with_values(self, values):
        return PointCloud(elements=self.elements, values=values, extrapolation=self.extrapolation, add_overlapping=self._add_overlapping, bounds=self._bounds, color=self._color)

    def with_extrapolation(self, extrapolation: Extrapolation):
        return PointCloud(elements=self.elements, values=self.values, extrapolation=extrapolation, add_overlapping=self._add_overlapping, bounds=self._bounds, color=self._color)

    def with_color(self, color: str or Tensor or tuple or list):
        return PointCloud(elements=self.elements, values=self.values, extrapolation=self.extrapolation, add_overlapping=self._add_overlapping, bounds=self._bounds, color=color)

    def with_bounds(self, bounds: Box):
        return PointCloud(elements=self.elements, values=self.values, extrapolation=self.extrapolation, add_overlapping=self._add_overlapping, bounds=bounds, color=self._color)

    def __value_attrs__(self):
        return '_values', '_extrapolation'

    def __variable_attrs__(self):
        return '_values', '_elements'

    def __eq__(self, other):
        if not type(self) == type(other):
            return False
        # Check everything but __variable_attrs__ (values): elements type, extrapolation, add_overlapping
        if type(self.elements) is not type(other.elements):
            return False
        if self.extrapolation != other.extrapolation:
            return False
        if self._add_overlapping != other._add_overlapping:
            return False
        if self.values is None:
            return other.values is None
        if other.values is None:
            return False
        if not math.all_available(self.values) or not math.all_available(other.values):  # tracers involved
            if math.all_available(self.values) != math.all_available(other.values):
                return False
            else:  # both tracers
                return self.values.shape == other.values.shape
        return bool((self.values == other.values).all)

    @property
    def bounds(self) -> Box:
        if self._bounds is not None:
            return self._bounds
        else:
            from phi.field._field_math import data_bounds
            bounds = data_bounds(self.elements.center)
            radius = math.max(self.elements.bounding_radius())
            return Box(bounds.lower - radius, bounds.upper + radius)

    @property
    def color(self) -> Tensor:
        return self._color

    def _sample(self, geometry: Geometry, scheme: Scheme) -> Tensor:
        if geometry == self.elements:
            return self.values
        elif isinstance(geometry, GridCell):
            return self.grid_scatter(geometry.bounds, geometry.resolution, scheme.outside_points)
        elif isinstance(geometry, GeometryStack):
            sampled = [self._sample(g, scheme=scheme) for g in geometry.geometries]
            return math.stack(sampled, geometry.geometries.shape)
        else:
            raise NotImplementedError()

    def grid_scatter(self, bounds: Box, resolution: math.Shape, outside_handling: str):
        """
        Approximately samples this field on a regular grid using math.scatter().

        Args:
            outside_handling: `str` passed to `phi.math.scatter()`.
            bounds: physical dimensions of the grid
            resolution: grid resolution

        Returns:
            `CenteredGrid`

        """
        closest_index = bounds.global_to_local(self.points) * resolution - 0.5
        mode = 'add' if self._add_overlapping else 'mean'
        base = math.zeros(resolution)
        if isinstance(self.extrapolation, math.extrapolation.ConstantExtrapolation):
            base += self.extrapolation.value
        scattered = math.scatter(base, closest_index, self.values, mode=mode, outside_handling=outside_handling)
        return scattered

    def mask(self):
        """
        Returns an equivalent `PointCloud` with `values=1` and `extrapolation=0`

        Returns:
            `PointCloud`
        """
        return PointCloud(self.elements, bounds=self.bounds, color=self.color)

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


def distribute_points(geometries: tuple or list or Geometry or float,
                      dim: Shape = instance('points'),
                      points_per_cell: int = 8,
                      color: str = None,
                      center: bool = False,
                      radius: float = None,
                      extrapolation: float or Extrapolation = math.NAN,
                      **domain) -> PointCloud:
    """
    Transforms `Geometry` objects into a PointCloud.

    Args:
        geometries: Geometry objects marking the cells which should contain points
        dim: Dimension along which the points are listed.
        points_per_cell: Number of points for each cell of `geometries`
        color (Optional): Color of PointCloud
        center: Set all points to the center of the grid cells.
        radius: Sphere radius.
        extrapolation: Extrapolation for the `PointCloud`, default `NaN` used for FLIP.

    Returns:
         PointCloud representation of `geometries`.
    """
    warnings.warn("distribute_points() is deprecated. Construct a PointCloud directly.", DeprecationWarning)
    from phi.field import CenteredGrid
    if isinstance(geometries, (tuple, list, Geometry)):
        from phi.geom import union
        geometries = union(geometries)
    geometries = CenteredGrid(geometries, extrapolation, **domain)
    initial_points = _distribute_points(geometries.values, dim, points_per_cell, center=center)
    if radius is None:
        from phi.field._field_math import data_bounds
        radius = math.mean(data_bounds(initial_points).size) * 0.005
    from phi.geom import Sphere
    return PointCloud(Sphere(initial_points, radius=radius), extrapolation=geometries.extrapolation, color=color, bounds=geometries.bounds)


def _distribute_points(mask: math.Tensor, dim: Shape, points_per_cell: int = 1, center: bool = False) -> math.Tensor:
    """
    Generates points (either uniformly distributed or at the cell centers) according to the given tensor mask.

    Args:
        mask: Tensor with nonzero values at the indices where particles should get generated.
        points_per_cell: Number of particles to generate at each marked index
        center: Set points to cell centers. If False, points will be distributed using a uniform
            distribution within each cell.

    Returns:
        A tensor containing the positions of the generated points.
    """
    indices = math.to_float(math.nonzero(mask, list_dim=dim))
    temp = []
    for _ in range(points_per_cell):
        if center:
            temp.append(indices + 0.5)
        else:
            temp.append(indices + (math.random_uniform(indices.shape)))
    return math.concat(temp, dim=dim)
