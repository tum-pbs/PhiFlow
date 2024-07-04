from numbers import Number
from typing import Union, List, Callable, Optional

from phi import math
from phi.geom import Geometry, Box, Point, UniformGrid, Mesh, sample_function
from phi.math import Shape, Tensor, instance, spatial, Solve, dual, si2d
from phi.math.extrapolation import Extrapolation, ConstantExtrapolation, PERIODIC
from phiml.math import unstack, channel, rename_dims, batch, extrapolation
from ._field import Field, FieldInitializer, as_boundary, slice_off_constant_faces
from phiml.math._tensors import may_vary_along


def resample(value: Union[Field, Geometry, Tensor, float, FieldInitializer], to: Union[Field, Geometry], keep_boundary=False, **kwargs):
    """
    Samples a `Field`, `Geometry` or value at the sample points of the field `to`.
    The result will approximate `value` on the data structure of `to`.
    Unlike `sample()`, this method returns a `Field` object, not a `Tensor`.

    Aliases:
        `value.at(to)`, (and the deprecated `value @ to`).

    See Also:
        `sample()`, `reduce_sample()`, `Field.at()`, [Resampling overview](https://tum-pbs.github.io/PhiFlow/Fields.html#resampling-fields).

    Args:
        value: Object containing values to resample.
            This can be
        to: `Field` (`CenteredGrid`, `StaggeredGrid` or `PointCloud`) object defining the sample points.
            The current values of `to` are ignored.
        keep_boundary: Only available if `self` is a `Field`.
            If True, the resampled field will inherit the extrapolation from `self` instead of `representation`.
            This can result in non-compatible value tensors for staggered grids where the tensor size depends on the extrapolation type.
        **kwargs: Sampling arguments, e.g. to specify the numerical scheme.
            By default, linear interpolation is used.
            Grids also support 6th order implicit sampling at mid-points.

    Returns:
        Field object of same type as `representation`

    Examples:
        >>> grid = CenteredGrid(x=64, y=32)
        >>> field.resample(Noise(), to=grid)
        CenteredGrid[(xˢ=64, yˢ=32), size=(x=64, y=32), extrapolation=float64 0.0]
        >>> field.resample(1, to=grid)
        CenteredGrid[(xˢ=64, yˢ=32), size=(x=64, y=32), extrapolation=float64 0.0]
        >>> field.resample(Box(x=1, y=2), to=grid)
        CenteredGrid[(xˢ=64, yˢ=32), size=(x=64, y=32), extrapolation=float64 0.0]
        >>> field.resample(grid, to=grid) == grid
        True
    """
    assert isinstance(to, (Field, Geometry)), f"'to' must be a Field or Geomoetry but got {to}"
    if not isinstance(value, (Field, Geometry, FieldInitializer)):
        return to.with_values(value)
    if isinstance(value, Field) and keep_boundary:
        extrap = value.extrapolation
    elif isinstance(to, Field) and not keep_boundary:
        extrap = to.extrapolation
    else:
        raise AssertionError(f"Boundary cannot be determined, keep_boundary={keep_boundary}, value: {type(value)}, to: {type(to)}")
    resampled = sample(value, to, at=to.sampled_at if isinstance(to, Field) else 'center', boundary=extrap, dot_face_normal=to.geometry, **kwargs)
    return Field(to.geometry if isinstance(to, Field) else to, resampled, extrap)


def reduce_sample(field: Union[Field, Geometry, FieldInitializer, Callable],
                  geometry: Geometry or Field or Tensor,
                  **kwargs) -> math.Tensor:
    """Alias for `sample()` with `dot_face_normal=field.geometry`."""
    can_reduce = dual(field.values) in geometry.shape
    at = 'face' if dual(geometry) else 'center'
    return sample(field, geometry, at, field.boundary, dot_face_normal=field.geometry if can_reduce else None, **kwargs)


def sample(field: Union[Field, Geometry, FieldInitializer, Callable],
           geometry: Geometry or Field or Tensor,
           at: str = 'center',
           boundary: Union[Extrapolation, Tensor, Number] = None,
           dot_face_normal: Optional[Geometry] = None,
           **kwargs) -> math.Tensor:
    """
    Computes the field value inside the volume of the (batched) `geometry`.

    The field value may be determined by integrating over the volume, sampling the central value or any other way.

    The batch dimensions of `geometry` are matched with this field.
    The `geometry` must not share any channel dimensions with this field.
    Spatial dimensions of `geometry` can be used to sample a grid of geometries.

    See Also:
        `Field.at()`, [Resampling overview](https://tum-pbs.github.io/PhiFlow/Fields.html#resampling-fields).

    Args:
        field: Source `Field` to sample.
        geometry: Single or batched `phi.geom.Geometry` or `Field` or location `Tensor`.
            When passing a `Field`, its `elements` are used as sample points.
            When passing a vector-valued `Tensor`, a `Point` geometry will be created.
        at: One of 'center', 'face', 'vertex'
        boundary: Target extrapolation.
        dot_face_normal: If not `None` and , `field` is a vector field and `at=='face'`, the dot product between sampled field vectors and the face normals is returned instead.
        **kwargs: Sampling arguments, e.g. to specify the numerical scheme.
            By default, linear interpolation is used.
            Grids also support 6th order implicit sampling at mid-points.

    Returns:
        Sampled values as a `phi.math.Tensor`
    """
    # --- Process args ---
    assert at in ['center', 'face', 'vertex']
    if at == 'face':
        assert boundary is not None, "boundaries must be given when sampling at faces"
    geometry = _get_geometry(geometry)
    boundary = as_boundary(boundary, geometry) if boundary is not None else None
    if dot_face_normal is True:
        dot_face_normal = geometry
    if isinstance(field, Geometry):
        from ._field_math import mask
        field = mask(field)
    if isinstance(field, FieldInitializer):
        values = field._sample(geometry, at, boundary, **kwargs)
        if dot_face_normal is not None and at == 'face' and channel(values):
            if _are_axis_aligned(dot_face_normal.face_normals):
                values = math.stack([values[{'vector': i, '~vector': i}] for i in range(geometry.spatial_rank)], dual(**geometry.shape['vector'].untyped_dict))
            else:
                raise NotImplementedError
        field = Field(geometry, values, boundary)
    if callable(field):
        values = sample_function(field, geometry, at, boundary)
        field = Field(geometry, values, boundary)
    # --- Resample ---
    assert isinstance(field, Field), f"field must be a Field, Geometry or initializer but got {type(field)}"
    if at == 'center':
        if field.is_centered and field.sampled_elements.shallow_equals(geometry):
            return field.values
        if field.is_grid and not field.is_staggered:
            return sample_grid_at_centers(field, geometry, **kwargs)
        elif field.is_grid and field.is_staggered:
            return sample_staggered_grid(field, geometry, **kwargs)
        elif field.is_mesh and field.is_staggered:
            return math.finite_mean(field.values, dual)  # ToDo weigh by face areas?
        elif field.is_mesh:
            return sample_mesh(field, geometry.center, **kwargs)
        else:
            return scatter_to_centers(field, geometry, **kwargs)
    elif at == 'face':
        if field.is_staggered and field.geometry.shallow_equals(geometry) and field.geometry.face_shape == geometry.face_shape and field.geometry.shallow_equals(dot_face_normal):
            return field.values
        elif dot_face_normal is not None and channel(field):
            if _are_axis_aligned(dot_face_normal.face_normals):
                components = unstack(field, field.shape.channel.name)
                faces = math.unstack(slice_off_constant_faces(geometry.faces, geometry.boundary_faces, boundary), dual)
                sampled = [sample(c, p, **kwargs) for c, p in zip(components, faces)]
                return math.stack(sampled, dual(dot_face_normal.face_shape))
            else:
                raise NotImplementedError
        elif field.is_grid and field.is_centered:
            return sample_grid_at_faces(field, geometry, boundary, **kwargs)
        elif field.is_grid and field.is_staggered:
            faces = math.unstack(slice_off_constant_faces(geometry.faces, geometry.boundary_faces, boundary), dual)
            sampled = [sample(field, face, **kwargs) for face in faces]
            return math.stack(sampled, dual(geometry.face_shape))
        elif field.is_mesh and field.is_centered and field.geometry.shallow_equals(geometry):
            return centroid_to_faces(field, boundary, **kwargs)
        elif field.is_mesh and field.is_staggered and field.geometry.shallow_equals(geometry):
            return field.with_boundary(boundary)
        else:
            return scatter_to_faces(field, geometry, boundary, **kwargs)
        # geom_ch = channel(geometry).without('vector')
        # assert all(dim not in field.shape for dim in geom_ch)
        # if geom_ch:
        #     sampled = [field._sample(p, **kwargs) for p in geometry.unstack(geom_ch.name)]
        #     return math.stack(sampled, geom_ch)
    elif at == 'vertex':
        raise NotImplementedError


def _get_geometry(geometry):
    if isinstance(geometry, Field):
        return geometry.geometry
    elif isinstance(geometry, Tensor) and 'vector' in geometry.shape:
        return Point(geometry)
    elif isinstance(geometry, Geometry):
        return geometry
    else:
        raise ValueError(f"A Geometry, Field or location Tensor is required but got {geometry}")


def _are_axis_aligned(normals: Tensor):
    return not spatial(normals) and not instance(normals) and math.sum(normals != 0) == normals.vector.size


def scatter_to_centers(self: Field, geometry: Geometry, soft=False, scatter=False, outside_handling='discard', balance=0.5) -> Tensor:
    if geometry == self.geometry:
        return self.values
    if self.extrapolation is PERIODIC:
        raise NotImplementedError("Periodic PointClouds not yet supported")
    if isinstance(geometry, UniformGrid) and scatter:
        assert not soft, "Cannot soft-sample when scatter=True"
        return grid_scatter(self, geometry.bounds, geometry.resolution, outside_handling)
    else:
        assert not isinstance(self._geometry, Point), "Cannot sample Point-like elements with scatter=False"
        if may_vary_along(self._values, instance(self._values) & spatial(self._values)):
            raise NotImplementedError("Non-scatter resampling not yet supported for varying values")
        idx0 = (instance(self._values) & spatial(self._values)).first_index()
        outside = self.boundary.value if isinstance(self.boundary, ConstantExtrapolation) else 0
        if soft:
            frac_inside = self.geometry.approximate_fraction_inside(geometry, balance)
            return frac_inside * self._values[idx0] + (1 - frac_inside) * outside
        else:
            return math.where(self.geometry.lies_inside(geometry.center), self._values[idx0], outside)


def scatter_to_faces(field: Field, geometry: Geometry, extrapolation: Extrapolation, **kwargs) -> Tensor:
    if isinstance(geometry, UniformGrid):
        sampled = {dim: scatter_to_centers(field, g, **kwargs) for dim, g in geometry.staggered_cells(extrapolation).items()}
        return math.stack(sampled, dual(geometry.face_shape))
    raise NotImplementedError


def grid_scatter(data: Field, bounds: Box, resolution: math.Shape, outside_handling: str, add_overlapping: bool = False):
    """
    Approximately samples this field on a regular grid using math.scatter().

    Args:
        outside_handling: `str` passed to `phi.math.scatter()`.
        bounds: physical dimensions of the grid
        resolution: grid resolution

    Returns:
        `CenteredGrid`
    """
    closest_index = bounds.global_to_local(data.points) * resolution - 0.5
    mode = 'add' if add_overlapping else 'mean'
    base = math.zeros(resolution)
    if isinstance(data.boundary, ConstantExtrapolation):
        base += data.boundary.value
    scattered = math.scatter(base, closest_index, data.values, mode=mode, outside_handling=outside_handling)
    return scattered


def sample_grid_at_centers(self: Field, geometry: Geometry, order=2, implicit: Solve = None) -> Tensor:
    if geometry == self.bounds:
        return math.mean(self.values, spatial(self))
    if isinstance(geometry, UniformGrid):
        if self.geometry == geometry:
            return self.values
        elif math.close(self.dx, geometry.size):
            if all([math.close(offset, geometry.half_size) or math.close(offset, 0) for offset in math.abs(self.bounds.lower - geometry.bounds.lower)]):
                dyadic_interpolated = _dyadic_interplate(self, geometry.resolution, geometry.bounds, order=order, implicit=implicit)
                if dyadic_interpolated is not NotImplemented:
                    return dyadic_interpolated
            if order != 2:
                raise NotImplementedError(f"Only 6th-order implicit and 2nd-order resampling supported but got order={order}")
            fast_resampled = _shift_resample(self, geometry.resolution, geometry.bounds)
            if fast_resampled is not NotImplemented:
                return fast_resampled
    points = geometry.center
    local_points = self.bounds.global_to_local(points) * self.resolution - 0.5
    resampled_values = math.grid_sample(self.values, local_points, self.extrapolation, bounds=self.bounds)
    from ._embed import FieldEmbedding
    if isinstance(self.extrapolation, FieldEmbedding):
        if isinstance(geometry, UniformGrid) and ((geometry.bounds.upper <= self.bounds.upper).all or (geometry.bounds.lower >= self.bounds.lower).all):
            # geometry is a subgrid of self
            return resampled_values
        else:  # otherwise we also sample the extrapolation Field
            ext_values = sample(self.extrapolation.field, geometry, order=order, implicit=implicit)
            inside = self.bounds.lies_inside(points)
            return math.where(inside, resampled_values, ext_values)
    return resampled_values


def sample_grid_at_faces(field: Field, elements: Geometry, extrapolation: Extrapolation, order=2, implicit: Solve = None):
    if isinstance(elements, UniformGrid):
        sampled = {dim: sample_grid_at_centers(field, g, order=order, implicit=implicit) for dim, g in elements.staggered_cells(extrapolation).items()}
        return math.stack(sampled, dual(elements.face_shape))
    raise NotImplementedError


def sample_staggered_grid(self: Field, geometry: Geometry, **kwargs) -> Tensor:
    components = []
    for dim in self.vector.item_names:
        c_values = self.values[{'~vector': dim}]
        c_grid = UniformGrid(self.resolution, self.bounds).stagger(dim, *self.extrapolation.valid_outer_faces(dim))
        ext = self.extrapolation[{'vector': dim}]
        c_sampled = sample_grid_at_centers(Field(c_grid, c_values, ext), geometry, **kwargs)
        components.append(c_sampled)
    return math.stack(components, geometry.shape['vector'])


def _dyadic_interplate(self: Field, resolution: Shape, bounds: Box, order=2, implicit: Solve = None):
    offsets = bounds.lower - self.bounds.lower
    interpolation_dirs = [0 if math.close(offset, 0) else int(math.sign(offset)) for offset in offsets]
    return _dyadic_interpolate(self.values, interpolation_dirs, self.extrapolation, order, implicit)


def _dyadic_interpolate(grid: Tensor, interpolation_dirs: List, padding: Extrapolation, order: int, implicit: Solve):
    """
    Samples a sub-grid from `grid` with an offset of half a grid cell in directions defined by `interpolation_dirs`.

    Args:
        grid: `Tensor` to be resampled.
        interpolation_dirs: List which defines for every spatial dimension of `grid` if interpolation should be performed,
            in positive direction `1` / negative direction `-1` / no interpolation`0`
            len(interpolation_dirs) == len(grid.shape.spatial.names) is assumed
            Example: With `grid.shape.spatial.names=['x', 'y']` and `interpolation_dirs: [1, -1]`
                     grid will be interpolated half a grid cell in positive x direction and half a grid cell in negative y direction
        padding: Extrapolation used for the needed out of Domain values
        order: finite difference `Scheme` used for interpolation

    Returns:
      Sub-grid as `Tensor`
    """
    if implicit:
        if order == 6:
            values, needed_shifts = [1 / 20, 3 / 4, 3 / 4, 1 / 20], (-1, 0, 1, 2)
            values_rhs, needed_shifts_rhs = [3 / 10, 1, 3 / 10], (-1, 0, 1)
        else:
            return NotImplemented
    else:
        return NotImplemented
    result = grid
    for dim, dir in zip(grid.shape.spatial.names, interpolation_dirs):
        if dir == 0: continue
        is_neg_dir = dir == -1
        current_widths = [abs(min(needed_shifts)) + is_neg_dir, max(needed_shifts) - is_neg_dir]
        padded = math.pad(result, {dim: tuple(current_widths)}, padding)
        shifted = math.shift(padded, needed_shifts, [dim], padding=None, stack_dim=None)
        result = sum([value * shift_ for value, shift_ in zip(values, shifted)])
        if implicit:
            implicit.x0 = result
            result = math.solve_linear(dyadic_interpolate_lhs, result, implicit, values_rhs=values_rhs, needed_shifts_rhs=needed_shifts_rhs, dim=dim, padding=padding)
    return result


@math.jit_compile_linear(auxiliary_args="values_rhs, needed_shifts_rhs")
def dyadic_interpolate_lhs(x, values_rhs, needed_shifts_rhs, dim, padding):
    shifted = math.shift(x, needed_shifts_rhs, stack_dim=None, dims=[dim], padding=padding)
    return sum([value * shift_ for value, shift_ in zip(values_rhs, shifted)])


def _shift_resample(self: Field, resolution: Shape, bounds: Box, threshold=1e-5, max_padding=20):
    assert math.all_available(bounds.lower, bounds.upper), "Shift resampling requires 'bounds' to be available."
    lower = math.to_int32(math.ceil(math.maximum(0, self.bounds.lower - bounds.lower) / self.dx - threshold))
    upper = math.to_int32(math.ceil(math.maximum(0, bounds.upper - self.bounds.upper) / self.dx - threshold))
    if math.close(*math.unstack(lower, batch)) and math.close(*math.unstack(upper, batch)):  # incompatible resolutions
        lower = math.unstack(lower, batch)[0]
        upper = math.unstack(upper, batch)[0]
    else:
        return NotImplemented
    total_padding = int(math.sum(lower) + math.sum(upper))
    if total_padding > max_padding and self.extrapolation.native_grid_sample_mode:
        return NotImplemented
    elif total_padding > 0:
        from phi.field import pad
        padded = pad(self, {dim: (int(lower[i]), int(upper[i])) for i, dim in enumerate(self.shape.spatial.names)})
        grid_box, grid_resolution, grid_values = padded.bounds, padded.resolution, padded.values
    else:
        grid_box, grid_resolution, grid_values = self.bounds, self.resolution, self.values
    origin_in_local = grid_box.global_to_local(bounds.lower) * grid_resolution
    if batch(origin_in_local):
        math.assert_close(*math.unstack(origin_in_local, batch), msg=f"sample_subgrid requires equal start but got varying values along batch dim {batch(origin_in_local)}")
        origin_in_local = math.unstack(origin_in_local, batch)[0]
    data = math.sample_subgrid(grid_values, origin_in_local, resolution)
    return data


def centroid_to_faces(u: Field, boundary: Extrapolation, order=2, upwind: Field = None, ignore_skew=False, gradient: Field = None):
    assert isinstance(upwind, Field) or upwind is None, f"upwind must be a Field but got {type(upwind)}"
    if '~neighbors' in u.values.shape:
        return u.values
    neighbor_val = u.mesh.pad_boundary(u.values, mode=u.boundary)
    upwind = upwind.at_faces(extrapolation.NONE, order=2, upwind=None) if upwind is not None else None
    if order == 2 and upwind is not None:  # linear upwind
        flows_out = upwind.values.vector @ u.mesh.face_normals.vector >= 0
        if gradient is None:
            from phi.field._field_math import green_gauss_gradient
            gradient = green_gauss_gradient(u, boundary=boundary, order=order, upwind=None, stack_dim=dual('vector'))  # we cannot pass same interpolation here
        neighbor_grad = u.mesh.pad_boundary(gradient.values, mode=boundary if boundary != extrapolation.NONE else u.boundary.spatial_gradient())
        interpolated_from_self = u.values + gradient.values.vector.dual @ (u.mesh.face_centers - u.mesh.center).vector
        interpolated_from_neighbor = neighbor_val + neighbor_grad.vector.dual @ (u.mesh.face_centers - (u.mesh.center + u.mesh.neighbor_offsets)).vector
        # ToDo limiter
        result = math.where(flows_out, interpolated_from_self, interpolated_from_neighbor)
        return slice_off_constant_faces(result, u.mesh.boundary_faces, boundary)
    elif order == 1 and upwind is not None:  # upwind (not linear)
        flows_out = upwind.values.vector @ u.mesh.face_normals.vector >= 0
        return math.where(flows_out, u.mesh.connectivity * u.values, u.mesh.connectivity * neighbor_val)
    elif order == 2:  # linear (not upwind)
        if ignore_skew:
            relative_face_distance = slice_off_constant_faces(u.mesh.relative_face_distance, u.mesh.boundary_faces, boundary)
            return (1 - relative_face_distance) * u.values + relative_face_distance * neighbor_val
        else:  # skew correction
            nb_center = math.replace_dims(u.mesh.center, 'cells', dual('~neighbors'))
            cell_deltas = math.pairwise_distances(u.mesh.center, format=u.mesh.cell_connectivity, default=None)  # x_N - x_P
            face_distance = nb_center - u.mesh.face_centers[u.mesh.interior_faces]  # x_N - x_f
            # face_distance = u.mesh.face_centers[u.mesh.interior_faces] - u.mesh.center  # x_f - x_P
            normals = u.mesh.face_normals[u.mesh.interior_faces]
            w_interior = (face_distance.vector @ normals.vector) / (cell_deltas.vector @ normals.vector)  # n·(x_N - x_f) / n·(x_N - x_P)
            w = math.concat([w_interior, 0 * u.mesh.boundary_connectivity], '~neighbors')
            w = slice_off_constant_faces(w, u.mesh.boundary_faces, boundary)  # first padding, then slicing is inefficient, but usually we don't slice anything off (boundary=none)
            # w = u.mesh.pad_boundary(w_interior, {k: s for k, s in u.mesh.boundary_faces.items() if not boundary.determines_boundary_values(k)}, boundary) this is only for vectors
            # b0 = math.tensor_like(slice_off_constant_faces(u.mesh.connectivity, u.mesh.boundary_faces, boundary), 0)
            return w * u.values + (1 - w) * neighbor_val
    else:
        raise NotImplementedError(f"resampling centroid to faces not supported for order={order}, upwind={upwind} not supported for resampling mesh values to faces")


def sample_mesh(f: Field,
                location: Tensor,
                gradient: Union[str, Field] = 'green-gauss',
                order=2,
                max_steps=None):  # at least 2 to resolve locations outside the mesh
    max_steps = f.mesh._max_cell_walk if max_steps is None else max_steps
    idx = math.find_closest(f.center, location)
    for i in range(max_steps):
        idx, leaves_mesh, is_outside, *_ = f.mesh.cell_walk_towards(location, idx, allow_exit=i == max_steps - 1)
    is_outside_mesh = leaves_mesh & is_outside
    if order <= 1:
        values = rename_dims(f.mesh.pad_boundary(f.values, mode=f.boundary), dual, instance(f))
        return values[idx]
    elif order == 2:
        values = rename_dims(f.mesh.pad_boundary(f.values, mode=f.boundary), dual, instance(f))
        v0 = values[idx]
        gradient = gradient if isinstance(gradient, Field) else f.gradient(scheme=gradient, order=order, stack_dim=f.geometry.shape['vector'].as_dual())
        grad = gradient.values[math.where(is_outside_mesh, 0, idx)]
        dx = location - f.mesh.center[math.where(is_outside_mesh, 0, idx)]
        return math.where(is_outside_mesh, v0, v0 + grad @ dx)
    raise NotImplementedError(f"sampling meshes only supports order <= 2 but got order={order}")
