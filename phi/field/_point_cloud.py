import warnings
from typing import Any, Tuple, Union

from phiml.math import wrap, expand, non_batch, extrapolation, spatial

from phi import math
from phi.geom import Geometry, GridCell, Box, Point
from ._field import SampledField, resample
from ..geom._stack import GeometryStack
from phiml.math import Tensor, instance, Shape
from phiml.math._tensors import may_vary_along
from phiml.math.extrapolation import Extrapolation, ConstantExtrapolation, PERIODIC
from phiml.math.magic import slicing_dict


class PointCloud(SampledField):
    """
    A `PointCloud` comprises:

    * `elements`: a `Geometry` representing all points or volumes
    * `values`: a `Tensor` representing the values corresponding to `elements`
    * `extrapolation`: an `Extrapolation` defining the field value outside of `values`

    The points / elements of the `PointCloud` are listed along *instance* or *spatial* dimensions of `elements`.
    These dimensions are automatically added to `values` if not already present.

    When sampling or resampling a `PointCloud`, the following keyword arguments can be specified.

    * `soft`: default=False.
      If `True`, interpolates smoothly from 1 to 0 between the inside and outside of elements.
      If `False`, only the center position of the new representation elements is checked against the point cloud elements.
    * `scatter`: default=False.
      If `True`, scattering will be used to sample the point cloud onto grids. Then, each element of the point cloud can only affect a single cell. This is only recommended when the points are much smaller than the cells.
    * `outside_handling`: default='discard'. One of `'discard'`, `'clamp'`, `'undefined'`.
    * `balance`: default=0.5. Only used when `soft=True`.
      See the description in `phi.geom.Geometry.approximate_fraction_inside()`.

    See the `phi.field` module documentation at https://tum-pbs.github.io/PhiFlow/Fields.html
    """

    def __init__(self,
                 elements: Union[Tensor, Geometry],
                 values: Any = 1.,
                 extrapolation: Union[Extrapolation, float] = 0.,
                 add_overlapping=False,
                 bounds: Box = None):
        """
        Args:
          elements: `Tensor` or `Geometry` object specifying the sample points and sizes
          values: values corresponding to elements
          extrapolation: values outside elements
          add_overlapping: True: values of overlapping geometries are summed. False: values between overlapping geometries are interpolated
          bounds: (optional) size of the fixed domain in which the points should get visualized. None results in max and min coordinates of points.
        """
        SampledField.__init__(self, elements, expand(wrap(values), non_batch(elements).non_channel), extrapolation, bounds)
        assert self._extrapolation is PERIODIC or isinstance(self._extrapolation, ConstantExtrapolation), f"Unsupported extrapolation for PointCloud: {self._extrapolation}"
        self._add_overlapping = add_overlapping

    @property
    def shape(self):
        return self._elements.shape.without('vector') & self._values.shape

    def __getitem__(self, item):
        item = slicing_dict(self, item)
        if not item:
            return self
        item_without_vec = {dim: selection for dim, selection in item.items() if dim != 'vector'}
        elements = self.elements[item_without_vec]
        values = self._values[item]
        extrapolation = self._extrapolation[item]
        bounds = self._bounds[item_without_vec] if self._bounds is not None else None
        return PointCloud(elements, values, extrapolation, self._add_overlapping, bounds)

    def with_elements(self, elements: Geometry):
        return PointCloud(elements=elements, values=self.values, extrapolation=self.extrapolation, add_overlapping=self._add_overlapping, bounds=self._bounds)

    def shifted(self, delta):
        return self.with_elements(self.elements.shifted(delta))

    def with_values(self, values):
        return PointCloud(elements=self.elements, values=values, extrapolation=self.extrapolation, add_overlapping=self._add_overlapping, bounds=self._bounds)

    def with_extrapolation(self, extrapolation: Extrapolation):
        return PointCloud(elements=self.elements, values=self.values, extrapolation=extrapolation, add_overlapping=self._add_overlapping, bounds=self._bounds)

    def with_bounds(self, bounds: Box):
        return PointCloud(elements=self.elements, values=self.values, extrapolation=self.extrapolation, add_overlapping=self._add_overlapping, bounds=bounds)

    def __value_attrs__(self):
        return '_values', '_extrapolation'

    def __variable_attrs__(self):
        return '_values', '_elements'

    def __expand__(self, dims: Shape, **kwargs) -> 'PointCloud':
        return self.with_values(expand(self.values, dims, **kwargs))

    def __replace_dims__(self, dims: Tuple[str, ...], new_dims: Shape, **kwargs) -> 'PointCloud':
        elements = math.rename_dims(self.elements, dims, new_dims)
        values = math.rename_dims(self.values, dims, new_dims)
        extrapolation = math.rename_dims(self.extrapolation, dims, new_dims, **kwargs)
        return PointCloud(elements, values, extrapolation, self._add_overlapping, self._bounds)

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

    def _sample(self, geometry: Geometry, soft=False, scatter=False, outside_handling='discard', balance=0.5) -> Tensor:
        if geometry == self.elements:
            return self.values
        if isinstance(geometry, GeometryStack):
            sampled = [self._sample(g, soft, scatter, outside_handling, balance) for g in geometry.geometries]
            return math.stack(sampled, geometry.geometries.shape)
        if self.extrapolation is extrapolation.PERIODIC:
            raise NotImplementedError("Periodic PointClouds not yet supported")
        if isinstance(geometry, GridCell) and scatter:
            assert not soft, "Cannot soft-sample when scatter=True"
            return self.grid_scatter(geometry.bounds, geometry.resolution, outside_handling)
        else:
            assert not isinstance(self._elements, Point), "Cannot sample Point-like elements with scatter=False"
            if may_vary_along(self._values, instance(self._values) & spatial(self._values)):
                raise NotImplementedError("Non-scatter resampling not yet supported for varying values")
            idx0 = (instance(self._values) & spatial(self._values)).first_index()
            outside = self._extrapolation.value if isinstance(self._extrapolation, ConstantExtrapolation) else 0
            if soft:
                frac_inside = self.elements.approximate_fraction_inside(geometry, balance)
                return frac_inside * self._values[idx0] + (1 - frac_inside) * outside
            else:
                return math.where(self.elements.lies_inside(geometry.center), self._values[idx0], outside)

    def grid_scatter(self, bounds: Box, resolution: math.Shape, outside_handling: str):
        """
        Approximately samples this field on a regular grid using math.scatter().

        Args:
            outside_handling: `str` passed to `phiml.math.scatter()`.
            bounds: physical dimensions of the grid
            resolution: grid resolution

        Returns:
            `CenteredGrid`
        """
        closest_index = bounds.global_to_local(self.points) * resolution - 0.5
        mode = 'add' if self._add_overlapping else 'mean'
        base = math.zeros(resolution)
        if isinstance(self._extrapolation, ConstantExtrapolation):
            base += self._extrapolation.value
        scattered = math.scatter(base, closest_index, self.values, mode=mode, outside_handling=outside_handling)
        return scattered

    def __repr__(self):
        try:
            return "PointCloud[%s]" % (self.shape,)
        except:
            return "PointCloud[invalid]"

    def __and__(self, other):
        assert isinstance(other, PointCloud)
        assert instance(self).rank == instance(other).rank == 1, f"Can only use & on PointClouds that have a single instance dimension but got shapes {self.shape} & {other.shape}"
        from ._field_math import concat
        return concat([self, other], instance(self))


def nonzero(field: SampledField):
    indices = math.nonzero(field.values, list_dim=instance('points'))
    elements = field.elements[indices]
    return PointCloud(elements, values=math.tensor(1.), extrapolation=math.extrapolation.ZERO, add_overlapping=False, bounds=field.bounds)


def distribute_points(geometries: Union[tuple, list, Geometry, float],
                      dim: Shape = instance('points'),
                      points_per_cell: int = 8,
                      center: bool = False,
                      radius: float = None,
                      extrapolation: Union[float, Extrapolation] = math.NAN,
                      **domain) -> PointCloud:
    """
    Transforms `Geometry` objects into a PointCloud.

    Args:
        geometries: Geometry objects marking the cells which should contain points
        dim: Dimension along which the points are listed.
        points_per_cell: Number of points for each cell of `geometries`
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
    geometries = resample(geometries, CenteredGrid(0, extrapolation, **domain), scatter=False)
    initial_points = _distribute_points(geometries.values, dim, points_per_cell, center=center)
    if radius is None:
        from phi.field._field_math import data_bounds
        radius = math.mean(data_bounds(initial_points).size) * 0.005
    from phi.geom import Sphere
    return PointCloud(Sphere(initial_points, radius=radius), extrapolation=geometries.extrapolation, bounds=geometries.bounds)


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
