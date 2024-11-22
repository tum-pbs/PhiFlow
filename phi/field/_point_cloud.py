import warnings
from typing import Any, Union

from phi import math, geom
from phi.geom import Geometry, Box
from phiml.math import shape
from ._field import Field
from ._resample import resample
from ..math import Tensor, instance, Shape, dual
from ..math.extrapolation import Extrapolation, ConstantExtrapolation, PERIODIC


def PointCloud(elements: Union[Tensor, Geometry, float], values: Any = 1., extrapolation: Union[Extrapolation, float] = 0., bounds: Box = None, variable_attrs=('values', 'geometry'), value_attrs=('values',)) -> Field:
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

    Args:
        elements: `Tensor` or `Geometry` object specifying the sample points and sizes
        values: values corresponding to elements
        extrapolation: values outside elements
        bounds: Deprecated. Has no use since 2.5.
    """
    if bounds is not None:
        warnings.warn("bounds argument is deprecated since 2.5 and will be ignored.", stacklevel=2)
    # if dual(values):
    #     assert dual(values).rank == 1, f"PointCloud cannot convert values with more than 1 dual dimension."
    #     non_dual_name = dual(values).name[1:]
    #     indices = math.stored_indices(values)[non_dual_name]
    #     values = math.stored_values(values)
    #     elements = elements[{non_dual_name: indices}]
    if isinstance(elements, (int, float)) and elements == 0:
        assert 'vector' in shape(values), f"When constructing a PointCloud from the origin 0, values must have a 'vector' dimension"
        elements = values * 0
    if isinstance(elements, Tensor):
        elements = geom.Point(elements)
    result = Field(elements, values, extrapolation, variable_attrs, value_attrs)
    assert result.boundary is PERIODIC or isinstance(result.boundary, ConstantExtrapolation), f"Unsupported extrapolation for PointCloud: {result._boundary}"
    return result


def nonzero(field: Field) -> Field:
    indices = math.nonzero(field.values, list_dim=instance('points'))
    elements = field.elements[indices]
    return PointCloud(elements, values=math.tensor(1.), extrapolation=math.extrapolation.ZERO, bounds=field.bounds)


def distribute_points(geometries: Union[tuple, list, Geometry, float],
                      dim: Shape = instance('points'),
                      points_per_cell: int = 8,
                      center: bool = False,
                      radius: float = None,
                      extrapolation: float or Extrapolation = math.NAN,
                      bounds: Box = None,
                      **domain) -> Field:
    """
    Transforms `Geometry` objects into a PointCloud.

    Args:
        geometries: Geometry objects marking the cells which should contain points
        dim: Dimension along which the points are listed.
        points_per_cell: Number of points for each cell of `geometries`
        center: Set all points to the center of the grid cells.
        radius: Sphere radius.
        extrapolation: Extrapolation for the `PointCloud`, default `NaN` used for FLIP.
        bounds: Grid bounds. Specify grid resolution via `**domain`.

    Returns:
         PointCloud representation of `geometries`.
    """
    from phi.field import CenteredGrid
    if isinstance(geometries, (tuple, list, Geometry)):
        from phi.geom import union
        geometries = union(geometries)
    geometries = resample(geometries, CenteredGrid(0, extrapolation, bounds=bounds, **domain), scatter=False)
    initial_points = _distribute_points(geometries.values, dim, points_per_cell, center=center)
    initial_points = geometries.bounds.local_to_global(initial_points / geometries.resolution)
    if radius is None:
        from phi.field._field_math import data_bounds
        radius = math.mean(data_bounds(initial_points).size) * 0.005
    from phi.geom import Sphere
    return PointCloud(Sphere(initial_points, radius=radius), extrapolation=geometries.extrapolation)


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
