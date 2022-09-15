from numbers import Number
from typing import Callable, List, Tuple

from phi import geom
from phi import math
from phi.geom import Box, Geometry, Sphere, Cuboid
from phi.math import Tensor, spatial, instance, tensor, masked_fill, channel, Shape, batch, unstack, wrap, vec
from ._field import Field, SampledField, SampledFieldType, as_extrapolation
from ._grid import CenteredGrid, Grid, StaggeredGrid, GridType
from ._point_cloud import PointCloud
from .numerical import Scheme
from ..math.extrapolation import Extrapolation


def bake_extrapolation(grid: GridType) -> GridType:
    """
    Pads `grid` with its current extrapolation.
    For `StaggeredGrid`s, the resulting grid will have a consistent shape, independent of the original extrapolation.

    Args:
        grid: `CenteredGrid` or `StaggeredGrid`.

    Returns:
        Padded grid with extrapolation `phi.math.extrapolation.NONE`.
    """
    if grid.extrapolation == math.extrapolation.NONE:
        return grid
    if isinstance(grid, StaggeredGrid):
        values = grid.values.unstack('vector')
        padded = []
        for dim, value in zip(grid.shape.spatial.names, values):
            lower, upper = grid.extrapolation.valid_outer_faces(dim)
            padded.append(math.pad(value, {dim: (0 if lower else 1, 0 if upper else 1)}, grid.extrapolation[{'vector': dim}], bounds=grid.bounds))
        return StaggeredGrid(math.stack(padded, grid.shape['vector']), bounds=grid.bounds, extrapolation=math.extrapolation.NONE)
    elif isinstance(grid, CenteredGrid):
        return pad(grid, 1).with_extrapolation(math.extrapolation.NONE)
    else:
        raise ValueError(f"Not a valid grid: {grid}")


def laplace(field: GridType, axes=spatial) -> GridType:
    """ Finite-difference laplace operator for Grids. See `phi.math.laplace()`. """
    result = field._op1(lambda tensor: math.laplace(tensor, dx=field.dx, padding=field.extrapolation, dims=axes))
    return result


def spatial_gradient(field: Field,
                     extrapolation: Extrapolation = None,
                     type: type = CenteredGrid,
                     stack_dim: Shape = channel('vector')):
    """
    Finite difference spatial_gradient.

    This function can operate in two modes:

    * `type=CenteredGrid` approximates the spatial_gradient at cell centers using central differences
    * `type=StaggeredGrid` computes the spatial_gradient at face centers of neighbouring cells

    Args:
        field: centered grid of any number of dimensions (scalar field, vector field, tensor field)
        type: either `CenteredGrid` or `StaggeredGrid`
        stack_dim: Dimension to be added. This dimension lists the spatial_gradient w.r.t. the spatial dimensions.
            The `field` must not have a dimension of the same name.

    Returns:
        spatial_gradient field of type `type`.

    """
    assert isinstance(field, Grid)
    if extrapolation is None:
        extrapolation = field.extrapolation.spatial_gradient()
    if type == CenteredGrid:
        values = math.spatial_gradient(field.values, field.dx.vector.as_channel(name=stack_dim.name), difference='central', padding=field.extrapolation, stack_dim=stack_dim)
        return CenteredGrid(values, bounds=field.bounds, extrapolation=extrapolation)
    elif type == StaggeredGrid:
        assert stack_dim.name == 'vector'
        return stagger(field, lambda lower, upper: (upper - lower) / field.dx, extrapolation)
    raise NotImplementedError(f"{type(field)} not supported. Only CenteredGrid and StaggeredGrid allowed.")


def shift(grid: CenteredGrid, offsets: tuple, stack_dim: Shape = channel('shift')):
    """
    Wraps :func:`math.shift` for CenteredGrid.

    Args:
      grid: CenteredGrid: 
      offsets: tuple: 
      stack_dim:  (Default value = 'shift')
    """
    data = math.shift(grid.values, offsets, padding=grid.extrapolation, stack_dim=stack_dim)
    return [CenteredGrid(data[i], bounds=grid.bounds, extrapolation=grid.extrapolation) for i in range(len(offsets))]


def stagger(field: CenteredGrid,
            face_function: Callable,
            extrapolation: float or math.extrapolation.Extrapolation,
            type: type = StaggeredGrid):
    """
    Creates a new grid by evaluating `face_function` given two neighbouring cells.
    One layer of missing cells is inferred from the extrapolation.
    
    This method returns a Field of type `type` which must be either StaggeredGrid or CenteredGrid.
    When returning a StaggeredGrid, the new values are sampled at the faces of neighbouring cells.
    When returning a CenteredGrid, the new grid has the same resolution as `field`.

    Args:
      field: centered grid
      face_function: function mapping (value1: Tensor, value2: Tensor) -> center_value: Tensor
      extrapolation: extrapolation mode of the returned grid. Has no effect on the values.
      type: one of (StaggeredGrid, CenteredGrid)
      field: CenteredGrid: 
      face_function: Callable:
      extrapolation: math.extrapolation.Extrapolation: 
      type: type:  (Default value = StaggeredGrid)

    Returns:
      grid of type matching the `type` argument

    """
    extrapolation = as_extrapolation(extrapolation)
    all_lower = []
    all_upper = []
    if type == StaggeredGrid:
        for dim in field.shape.spatial.names:
            lo_valid, up_valid = extrapolation.valid_outer_faces(dim)
            width_lower = {dim: (int(lo_valid), int(up_valid) - 1)}
            width_upper = {dim: (int(lo_valid or up_valid) - 1, int(lo_valid and up_valid))}
            all_lower.append(math.pad(field.values, width_lower, field.extrapolation, bounds=field.bounds))
            all_upper.append(math.pad(field.values, width_upper, field.extrapolation, bounds=field.bounds))
        all_upper = math.stack(all_upper, channel('vector'))
        all_lower = math.stack(all_lower, channel('vector'))
        values = face_function(all_lower, all_upper)
        result = StaggeredGrid(values, bounds=field.bounds, extrapolation=extrapolation)
        assert result.shape.spatial == field.shape.spatial
        return result
    elif type == CenteredGrid:
        left, right = math.shift(field.values, (-1, 1), padding=field.extrapolation, stack_dim=channel('vector'))
        values = face_function(left, right)
        return CenteredGrid(values, bounds=field.bounds, extrapolation=extrapolation)
    else:
        raise ValueError(type)


def divergence(field: Grid) -> CenteredGrid:
    """
    Computes the divergence of a grid using finite differences.

    This function can operate in two modes depending on the type of `field`:

    * `CenteredGrid` approximates the divergence at cell centers using central differences
    * `StaggeredGrid` exactly computes the divergence at cell centers

    Args:
        field: vector field as `CenteredGrid` or `StaggeredGrid`

    Returns:
        Divergence field as `CenteredGrid`
    """
    if isinstance(field, StaggeredGrid):
        field = bake_extrapolation(field)
        components = []
        for dim in field.shape.spatial.names:
            div_dim = math.spatial_gradient(field.values.vector[dim], field.dx, 'forward', None, dims=dim, stack_dim=None)
            components.append(div_dim)
        data = math.sum(components, dim='0')
        return CenteredGrid(data, bounds=field.bounds, extrapolation=field.extrapolation.spatial_gradient())
    elif isinstance(field, CenteredGrid):
        left, right = shift(field, (-1, 1), stack_dim=batch('div_'))
        grad = (right - left) / (field.dx * 2)
        components = [grad.vector[i].div_[i] for i in range(grad.div_.size)]
        result = sum(components)
        return result
    else:
        raise NotImplementedError(f"{type(field)} not supported. Only StaggeredGrid allowed.")


def curl(field: Grid, type: type = CenteredGrid):
    """ Computes the finite-difference curl of the give 2D `StaggeredGrid`. """
    assert field.spatial_rank in (2, 3), "curl is only defined in 2 and 3 spatial dimensions."
    if isinstance(field, CenteredGrid) and field.spatial_rank == 2:
        if 'vector' not in field.shape and type == StaggeredGrid:
            # 2D curl of scalar field
            grad = math.spatial_gradient(field.values, dx=field.dx, difference='forward', padding=None, stack_dim=channel('vector'))
            result = grad.vector.flip() * (1, -1)  # (d/dy, -d/dx)
            bounds = Box(field.bounds.lower + 0.5 * field.dx, field.bounds.upper - 0.5 * field.dx)  # lose 1 cell per dimension
            return StaggeredGrid(result, bounds=bounds, extrapolation=field.extrapolation.spatial_gradient())
        if 'vector' in field.shape and type == CenteredGrid:
            # 2D curl of vector field
            x, y = field.shape.spatial.names
            vy_dx = math.spatial_gradient(field.values.vector[1], dx=field.dx.vector[0], padding=field.extrapolation, dims=x, stack_dim=None)
            vx_dy = math.spatial_gradient(field.values.vector[0], dx=field.dx.vector[1], padding=field.extrapolation, dims=y, stack_dim=None)
            c = vy_dx - vx_dy
            return field.with_values(c)
    elif isinstance(field, StaggeredGrid) and field.spatial_rank == 2:
        if type == CenteredGrid:
            for dim in field.resolution.names:
                l, u = field.extrapolation.valid_outer_faces(dim)
                assert l == u, "periodic extrapolation not yet supported"
            values = bake_extrapolation(field).values
            x_padded = math.pad(values.vector['x'], {'y': (1, 1)}, field.extrapolation)
            y_padded = math.pad(values.vector['y'], {'x': (1, 1)}, field.extrapolation)
            vx_dy = math.spatial_gradient(x_padded, field.dx, 'forward', None, dims='y', stack_dim=None)
            vy_dx = math.spatial_gradient(y_padded, field.dx, 'forward', None, dims='x', stack_dim=None)
            result = vy_dx - vx_dy
            return CenteredGrid(result, field.extrapolation.spatial_gradient(), bounds=field.bounds)
    raise NotImplementedError()


def fourier_laplace(grid: GridType, times=1) -> GridType:
    """ See `phi.math.fourier_laplace()` """
    assert grid.extrapolation.spatial_gradient() == math.extrapolation.PERIODIC
    values = math.fourier_laplace(grid.values, dx=grid.dx, times=times)
    return type(grid)(values=values, bounds=grid.bounds, extrapolation=grid.extrapolation)


def fourier_poisson(grid: GridType, times=1) -> GridType:
    """ See `phi.math.fourier_poisson()` """
    assert grid.extrapolation.spatial_gradient() == math.extrapolation.PERIODIC
    values = math.fourier_poisson(grid.values, dx=grid.dx, times=times)
    return type(grid)(values=values, bounds=grid.bounds, extrapolation=grid.extrapolation)


def native_call(f, *inputs, channels_last=None, channel_dim='vector', extrapolation=None) -> SampledField or Tensor:
    """
    Similar to `phi.math.native_call()`.

    Args:
        f: Function to be called on native tensors of `inputs.values`.
            The function output must have the same dimension layout as the inputs and the batch size must be identical.
        *inputs: `SampledField` or `phi.Tensor` instances.
        extrapolation: (Optional) Extrapolation of the output field. If `None`, uses the extrapolation of the first input field.

    Returns:
        `SampledField` matching the first `SampledField` in `inputs`.
    """
    input_tensors = [i.values if isinstance(i, SampledField) else tensor(i) for i in inputs]
    values = math.native_call(f, *input_tensors, channels_last=channels_last, channel_dim=channel_dim)
    for i in inputs:
        if isinstance(i, SampledField):
            result = i.with_values(values=values)
            if extrapolation is not None:
                result = result.with_extrapolation(extrapolation)
            return result
    else:
        raise AssertionError("At least one input must be a SampledField.")


def data_bounds(loc: SampledField or Tensor) -> Box:
    if isinstance(loc, SampledField):
        loc = loc.points
    assert isinstance(loc, Tensor), f"loc must be a Tensor or SampledField but got {type(loc)}"
    min_vec = math.min(loc, dim=loc.shape.non_batch.non_channel)
    max_vec = math.max(loc, dim=loc.shape.non_batch.non_channel)
    return Box(min_vec, max_vec)


def mean(field: SampledField) -> Tensor:
    """
    Computes the mean value by reducing all spatial / instance dimensions.

    Args:
        field: `SampledField`

    Returns:
        `phi.Tensor`
    """
    return math.mean(field.values, field.shape.non_channel.non_batch)


def normalize(field: SampledField, norm: SampledField, epsilon=1e-5):
    """ Multiplies the values of `field` so that its sum matches the source. """
    data = math.normalize_to(field.values, norm.values, epsilon)
    return field.with_values(data)


def center_of_mass(density: SampledField):
    """
    Compute the center of mass of a density field.

    Args:
        density: Scalar `SampledField`

    Returns:
        `Tensor` holding only batch dimensions.
    """
    assert 'vector' not in density.shape
    return mean(density.points * density) / mean(density)


def pad(grid: GridType, widths: int or tuple or list or dict) -> GridType:
    """
    Pads a `Grid` using its extrapolation.

    Unlike `phi.math.pad()`, this function also affects the `bounds` of the grid, changing its size and origin depending on `widths`.

    Args:
        grid: `CenteredGrid` or `StaggeredGrid`
        widths: Either `int` or `(lower, upper)` to pad the same number of cells in all spatial dimensions
            or `dict` mapping dimension names to `(lower, upper)`.

    Returns:
        `Grid` of the same type as `grid`
    """
    if isinstance(widths, int):
        widths = {axis: (widths, widths) for axis in grid.shape.spatial.names}
    elif isinstance(widths, (tuple, list)):
        widths = {axis: (width if isinstance(width, (tuple, list)) else (width, width)) for axis, width in zip(grid.shape.spatial.names, widths)}
    else:
        assert isinstance(widths, dict)
    widths_list = [widths[axis] for axis in grid.shape.spatial.names]
    if isinstance(grid, Grid):
        data = math.pad(grid.values, widths, grid.extrapolation, bounds=grid.bounds)
        w_lower = math.wrap([w[0] for w in widths_list])
        w_upper = math.wrap([w[1] for w in widths_list])
        bounds = Box(grid.box.lower - w_lower * grid.dx, grid.box.upper + w_upper * grid.dx)
        return type(grid)(values=data, bounds=bounds, extrapolation=grid.extrapolation)
    raise NotImplementedError(f"{type(grid)} not supported. Only Grid instances allowed.")


def downsample2x(grid: Grid) -> GridType:
    """
    Reduces the number of sample points by a factor of 2 in each spatial dimension.
    The new values are determined via linear interpolation.

    See Also:
        `upsample2x()`.

    Args:
        grid: `CenteredGrid` or `StaggeredGrid`.

    Returns:
        `Grid` of same type as `grid`.
    """
    if isinstance(grid, CenteredGrid):
        values = math.downsample2x(grid.values, grid.extrapolation)
        return CenteredGrid(values, bounds=grid.bounds, extrapolation=grid.extrapolation)
    elif isinstance(grid, StaggeredGrid):
        values = []
        for dim, centered_grid in zip(grid.shape.spatial.names, unstack(grid, 'vector')):
            odd_discarded = centered_grid.values[{dim: slice(None, None, 2)}]
            others_interpolated = math.downsample2x(odd_discarded, grid.extrapolation, dims=grid.shape.spatial.without(dim))
            values.append(others_interpolated)
        return StaggeredGrid(math.stack(values, channel('vector')), bounds=grid.bounds, extrapolation=grid.extrapolation)
    else:
        raise ValueError(type(grid))


def upsample2x(grid: GridType) -> GridType:
    """
    Increases the number of sample points by a factor of 2 in each spatial dimension.
    The new values are determined via linear interpolation.

    See Also:
        `downsample2x()`.

    Args:
        grid: `CenteredGrid` or `StaggeredGrid`.

    Returns:
        `Grid` of same type as `grid`.
    """
    if isinstance(grid, CenteredGrid):
        values = math.upsample2x(grid.values, grid.extrapolation)
        return CenteredGrid(values, bounds=grid.bounds, extrapolation=grid.extrapolation)
    elif isinstance(grid, StaggeredGrid):
        raise NotImplementedError()
    else:
        raise ValueError(type(grid))


def concat(fields: List[SampledFieldType] or Tuple[SampledFieldType, ...], dim: str or Shape) -> SampledFieldType:
    """
    Concatenates the given `SampledField`s along `dim`.

    See Also:
        `stack()`.

    Args:
        fields: List of matching `SampledField` instances.
        dim: Concatenation dimension as `Shape`. Size is ignored.

    Returns:
        `SampledField` matching concatenated fields.
    """
    assert all(isinstance(f, SampledField) for f in fields)
    assert all(isinstance(f, type(fields[0])) for f in fields)
    if any(f.extrapolation != fields[0].extrapolation for f in fields):
        raise NotImplementedError("Concatenating extrapolations not supported")
    if isinstance(fields[0], Grid):
        values = math.concat([f.values for f in fields], dim)
        return fields[0].with_values(values)
    elif isinstance(fields[0], PointCloud):
        elements = geom.concat([f.elements for f in fields], dim, sizes=[f.shape.get_size(dim) for f in fields])
        values = math.concat([math.expand(f.values, f.shape.only(dim)) for f in fields], dim)
        colors = math.concat([math.expand(f.color, f.shape.only(dim)) for f in fields], dim)
        return PointCloud(elements=elements, values=values, color=colors, extrapolation=fields[0].extrapolation, add_overlapping=fields[0]._add_overlapping, bounds=fields[0]._bounds)
    raise NotImplementedError(type(fields[0]))


def stack(fields, dim: Shape, dim_bounds: Box = None):
    """
    Stacks the given `SampledField`s along `dim`.

    See Also:
        `concat()`.

    Args:
        fields: List of matching `SampledField` instances.
        dim: Stack dimension as `Shape`. Size is ignored.

    Returns:
        `SampledField` matching stacked fields.
    """
    assert all(isinstance(f, SampledField) for f in fields), f"All fields must be SampledFields of the same type but got {fields}"
    assert all(isinstance(f, type(fields[0])) for f in fields), f"All fields must be SampledFields of the same type but got {fields}"
    if any(f.extrapolation != fields[0].extrapolation for f in fields):
        raise NotImplementedError("Concatenating extrapolations not supported")
    if isinstance(fields[0], Grid):
        values = math.stack([f.values for f in fields], dim)
        if spatial(dim):
            if dim_bounds is None:
                dim_bounds = Box(**{dim.name: len(fields)})
            return type(fields[0])(values, extrapolation=fields[0].extrapolation, bounds=fields[0].bounds * dim_bounds)
        else:
            return fields[0].with_values(values)
    elif isinstance(fields[0], PointCloud):
        elements = geom.stack([f.elements for f in fields], dim=dim)
        values = math.stack([f.values for f in fields], dim=dim)
        colors = math.stack([f.color for f in fields], dim=dim)
        return PointCloud(elements=elements, values=values, color=colors, extrapolation=fields[0].extrapolation, add_overlapping=fields[0]._add_overlapping, bounds=fields[0]._bounds)
    raise NotImplementedError(type(fields[0]))


def assert_close(*fields: SampledField or Tensor or Number,
                 rel_tolerance: float = 1e-5,
                 abs_tolerance: float = 0,
                 msg: str = "",
                 verbose: bool = True):
    """ Raises an AssertionError if the `values` of the given fields are not close. See `phi.math.assert_close()`. """
    f0 = next(filter(lambda t: isinstance(t, SampledField), fields))
    values = [(f @ f0).values if isinstance(f, SampledField) else math.wrap(f) for f in fields]
    math.assert_close(*values, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance, msg=msg, verbose=verbose)


def where(mask: Field or Geometry or float, field_true: Field or float, field_false: Field or float) -> SampledFieldType:
    """
    Element-wise where operation.
    Picks the value of `field_true` where `mask=1 / True` and the value of `field_false` where `mask=0 / False`.

    The fields are automatically resampled if necessary, preferring the sample points of `mask`.
    At least one of the arguments must be a `SampledField`.

    Args:
        mask: `Field` or `Geometry` object.
        field_true: `Field`
        field_false: `Field`

    Returns:
        `SampledField`
    """
    field_true, field_false, mask = _auto_resample(field_true, field_false, mask)
    values = math.where(mask.values, field_true.values, field_false.values)
    return field_true.with_values(values)


def maximum(f1: Field or Geometry or float, f2: Field or Geometry or float):
    """
    Element-wise maximum.
    One of the given fields needs to be an instance of `SampledField` and the the result will be sampled at the corresponding points.
    If both are `SampledFields` but have different points, `f1` takes priority.

    Args:
        f1: `Field` or `Geometry` or constant.
        f2: `Field` or `Geometry` or constant.

    Returns:
        `SampledField`
    """
    f1, f2 = _auto_resample(f1, f2)
    return f1.with_values(math.maximum(f1.values, f2.values))


def minimum(f1: Field or Geometry or float, f2: Field or Geometry or float):
    """
    Element-wise minimum.
    One of the given fields needs to be an instance of `SampledField` and the the result will be sampled at the corresponding points.
    If both are `SampledFields` but have different points, `f1` takes priority.

    Args:
        f1: `Field` or `Geometry` or constant.
        f2: `Field` or `Geometry` or constant.

    Returns:
        `SampledField`
    """
    f1, f2 = _auto_resample(f1, f2)
    return f1.with_values(math.minimum(f1.values, f2.values))


def _auto_resample(*fields: Field):
    """ Prefers extrapolation from first SampledField """
    for sampled_field in fields:
        if isinstance(sampled_field, SampledField):
            return [f @ sampled_field for f in fields]
    raise AssertionError(f"At least one argument must be a SampledField but got {fields}")


def vec_length(field: SampledField):
    """ See `phi.math.vec_abs()` """
    if isinstance(field, StaggeredGrid):
        field = field.at_centers()
    return field.with_values(math.vec_abs(field.values))


def vec_squared(field: SampledField):
    """ See `phi.math.vec_squared()` """
    if isinstance(field, StaggeredGrid):
        field = field.at_centers()
    return field.with_values(math.vec_squared(field.values))


def finite_fill(grid: GridType, distance=1, diagonal=True) -> GridType:
    """
    Extrapolates values of `grid` which are marked by nonzero values in `valid` using `phi.math.masked_fill().
    If `values` is a StaggeredGrid, its components get extrapolated independently.

    Args:
        grid: Grid holding the values for extrapolation and possible non-finite values to be filled.
        distance: Number of extrapolation steps, i.e. how far a cell can be from the closest finite value to get filled.
        diagonal: Whether to extrapolate values to their diagonal neighbors per step.

    Returns:
        grid: Grid with extrapolated values.
        valid: binary Grid marking all valid values after extrapolation.
    """
    if isinstance(grid, CenteredGrid):
        new_values = math.finite_fill(grid.values, distance=distance, diagonal=diagonal, padding=grid.extrapolation)
        return grid.with_values(new_values)
    elif isinstance(grid, StaggeredGrid):
        new_values = [finite_fill(c, distance=distance, diagonal=diagonal).values for c in grid.vector]
        return grid.with_values(math.stack(new_values, channel(grid)))
    else:
        raise ValueError(grid)


def discretize(grid: Grid, filled_fraction=0.25):
    """ Treats channel dimensions as batch dimensions. """
    import numpy as np
    data = math.reshaped_native(grid.values, [grid.shape.non_spatial, grid.shape.spatial])
    ranked_idx = np.argsort(data, axis=-1)
    filled_idx = ranked_idx[:, int(round(grid.shape.spatial.volume * (1 - filled_fraction))):]
    filled = np.zeros_like(data)
    np.put_along_axis(filled, filled_idx, 1, axis=-1)
    filled_t = math.reshaped_tensor(filled, [grid.shape.non_spatial, grid.shape.spatial])
    return grid.with_values(filled_t)


def integrate(field: Field, region: Geometry, scheme: Scheme = Scheme()) -> Tensor:
    """
    Computes *∫<sub>R</sub> f(x) dx<sup>d</sup>* , where *f* denotes the `Field`, *R* the `region` and *d* the number of spatial dimensions (`d=field.shape.spatial_rank`).
    Depending on the `sample` implementation for `field`, the integral may be a rough approximation.

    This method is currently only implemented for `CenteredGrid`.

    Args:
        field: `Field` to integrate.
        region: Region to integrate over.
        scheme: Numerical scheme.

    Returns:
        Integral as `phi.Tensor`
    """
    if not isinstance(field, CenteredGrid):
        raise NotImplementedError()
    return field._sample(region, scheme=scheme) * region.volume


def tensor_as_field(t: Tensor):
    """
    Interpret a `Tensor` as a `CenteredGrid` or `PointCloud` depending on its dimensions.

    Unlike the `CenteredGrid` constructor, this function will have the values sampled at integer points for each spatial dimension.

    Args:
        t: `Tensor` with either `spatial` or `instance` dimensions.

    Returns:
        `CenteredGrid` or `PointCloud`
    """
    if instance(t):
        bounds = data_bounds(t)
        return PointCloud(t, bounds=Cuboid(bounds.center, bounds.half_size * 1.2).box())
    elif spatial(t):
        return CenteredGrid(t, 0, bounds=Box(math.const_vec(-0.5, spatial(t)), wrap(spatial(t), channel('vector')) - 0.5))
    elif 'vector' in t.shape:
        return PointCloud(math.expand(t, instance(points=1)), bounds=Cuboid(t, half_size=math.const_vec(1, t.shape['vector'])).box())
    else:
        raise ValueError(f"Cannot create field from tensor with shape {t.shape}. Requires at least one spatial, instance or vector dimension.")


def pack_dims(field: SampledFieldType,
              dims: Shape or tuple or list or str,
              packed_dim: Shape,
              pos: int or None = None) -> SampledFieldType:
    """
    Currently only supports grids and non-spatial dimensions.

    See Also:
        `phi.math.pack_dims()`.

    Args:
        field: `SampledField`

    Returns:
        `SampledField` of same type as `field`.
    """
    if isinstance(field, Grid):
        if spatial(field.shape.only(dims)):
            raise NotImplementedError("Packing spatial dimensions not supported for grids")
        return field.with_values(math.pack_dims(field.values, dims, packed_dim, pos))
    else:
        raise NotImplementedError()


def support(field: SampledField, list_dim: Shape or str = instance('nonzero')) -> Tensor:
    """
    Returns the points at which the field values are non-zero.

    Args:
        field: `SampledField`
        list_dim: Dimension to list the non-zero values.

    Returns:
        `Tensor` with shape `(list_dim, vector)`
    """
    return field.points[math.nonzero(field.values, list_dim=list_dim)]
