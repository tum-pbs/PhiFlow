from numbers import Number
from typing import Callable, List, Tuple, Optional, Union, Sequence

import numpy as np
from phiml.math import Tensor, spatial, instance, tensor, channel, batch, Shape, unstack, solve_linear, \
    jit_compile_linear, \
    shape, Solve, extrapolation, dual, wrap, rename_dims, factorial, concat, zeros, ones, neighbor_mean
from phi import geom
from phi import math
from phi.geom import Box, Geometry, UniformGrid
from phiml.math._shape import auto, DimFilter
from phiml.math.extrapolation import NONE, domain_slice
from ._field import Field, as_boundary, slice_off_constant_faces
from ._grid import CenteredGrid, StaggeredGrid, grid, unstack_staggered_tensor
from ._point_cloud import PointCloud
from ._resample import sample
from ..math.extrapolation import Extrapolation, SYMMETRIC, REFLECT, ANTIREFLECT, ANTISYMMETRIC, combine_by_direction


def bake_extrapolation(grid: Field) -> Field:
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
    if grid.is_grid and grid.is_staggered:
        padded = []
        for dim in grid.vector.item_names:
            lower, upper = grid.extrapolation.valid_outer_faces(dim)
            value = grid.vector[dim].values
            padded.append(math.pad(value, {dim: (0 if lower else 1, 0 if upper else 1)}, grid.extrapolation[{'vector': dim}], bounds=grid.bounds))
        return StaggeredGrid(math.stack(padded, dual(vector=grid.shape.spatial)), bounds=grid.bounds, extrapolation=math.extrapolation.NONE)
    elif grid.is_grid:
        return pad(grid, 1).with_extrapolation(math.extrapolation.NONE)
    else:
        raise ValueError(f"Not a valid grid: {grid}")


def laplace(u: Field,
            axes: DimFilter = spatial,
            gradient: Field = None,
            order=2,
            implicit: math.Solve = None,
            implicitness: int = None,
            weights: Union[Tensor, Field] = None,
            upwind: Field = None,
            correct_skew=True,
            wide_stencil=False) -> Field:
    """
    Spatial Laplace operator for scalar grid.

    For grids, uses a finite difference scheme specified by `order` and `implicit`.
    For unstructured meshes, the scheme is specified via `order` and `upwind`.

    Args:
        u: n-dimensional grid or mesh.
        axes: The second derivative along these dimensions is summed over
        weights: (Optional) Multiply the axis terms by these factors before summation.
            Must be a `phi.math.Tensor` or `phi.field.Field` with a single channel dimension that lists all laplace axes by name.
        gradient: Only used by FVM at the moment. Approximate gradient of `u`, e.g. ∇u of the previous time step.
            If `None`, approximates the gradient as `(u_neighbor - u_self) / distance`.
        order: Spatial order of accuracy.
            Higher orders entail larger stencils and more computation time but result in more accurate results assuming a large enough resolution.
            Supported: 2 explicit, 4 explicit, 6 implicit (inherited from `phi.field.laplace()`).
            For FVM, the order is used when interpolating `v` and `prev_v` to cell faces if needed.
        implicit: When a `Solve` object is passed, performs an implicit operation with the specified solver and tolerances.
            Otherwise, an explicit stencil is used.
        implicitness: specifies the size of the implicit stencil in case an implicit treatment is used
        upwind: FVM only. Whether to use upwind interpolation.
        correct_skew: If `True`, adds a correction term for cell skewness. This requires `gradient` to be passed.

    Returns:
        laplacian field as `CenteredGrid`
    """
    if implicitness is None:
        implicitness = 0 if implicit is None else 2
    elif implicitness != 0:
        assert implicit is not None, "for implicit treatment a `Solve` is required"
    axes_names = u.shape.only(axes).names
    if isinstance(weights, Field):
        weights = weights.at(u).values
    if weights is not None:
        if channel(weights):
            assert set(channel(weights).item_names[0]) >= set(axes_names), f"the channel dim of weights must contain all laplace dims {axes_names} but only has {channel(weights).item_names}"
    # --- Mesh ---
    if u.is_mesh:
        if weights is not None and 'vector' in shape(weights):
            raise NotImplementedError(f"laplace on meshes is not yet supported with vector-valued weights")
        neighbor_val = u.mesh.pad_boundary(u.values, mode=u.boundary)
        nb_distances = u.mesh.neighbor_distances
        if wide_stencil:
            assert weights is None
            grad_p = spatial_gradient(u, order=order, scheme='green-gauss', upwind=upwind)
            div_grad_p = grad_p.divergence(order=order, upwind=upwind)
            return div_grad_p
        connecting_grad = (u.mesh.connectivity * neighbor_val - u.values) / nb_distances  # (T_N - T_P) / d_PN
        if correct_skew and gradient is not None:  # skewness correction
            assert dual(gradient).names == ('~vector',), f"gradient must contain one dual dim '~vector' listing the gradient components but got {gradient.shape}"
            gradient = gradient.at_faces(boundary=NONE, order=order, upwind=upwind).values
            nb_offsets = u.mesh.neighbor_offsets
            n1 = (u.face_normals.vector @ nb_offsets.vector) * nb_offsets / nb_distances ** 2  # (n·d_PN) d_PN / d_PN^2
            n2 = u.face_normals - n1
            ortho_correction = gradient @ n2
            grad = connecting_grad * math.vec_length(n1) + ortho_correction
        else:
            assert not correct_skew, f"FVM skew correction only available when gradient is specified. Pass gradient or set correct_skew=False"
            grad = connecting_grad
        laplace_values = u.mesh.integrate_surface(grad) / u.mesh.volume  # 1/V ∑_f ∇T ν A
        result = weights * laplace_values if weights is not None else laplace_values
        return Field(u.mesh, result, u.boundary - u.boundary)
    # --- Grid ---
    laplace_ext = u.extrapolation.spatial_gradient().spatial_gradient()
    laplace_dims = u.shape.only(axes).names
    if 'vector' in u.shape and (u.is_centered or order > 2):
        fields = [f for f in u.vector]
    else:
        fields = [u]
    result = []
    for f in fields:
        if order == 2:
            result.append(math.map_d2c(math.laplace)(f.values, dx=f.dx, padding=f.extrapolation, dims=axes, weights=weights, padding_kwargs={'bounds': f.bounds}))  # uses ghost cells
        else:
            result_components = [perform_finite_difference_operation(f.values, dim, 2, f.dx.vector[dim], f.extrapolation, laplace_ext, 'center', order, implicit, implicitness) for dim in laplace_dims]
            if weights is not None:
                if channel(weights):
                    result_components = [c * weights[ax] for c, ax in zip(result_components, axes_names)]
                else:
                    result_components = [c * weights for c in result_components]

            result.append(sum(result_components))
    if 'vector' in u.shape and (u.is_centered or order > 2):
        if u.is_staggered:
            result = math.stack(result, dual(vector=u.vector.item_names))
        else:
            result = math.stack(result, channel(vector=u.vector.item_names))
    else:
        result = result[0]
    return u.with_values(result).with_extrapolation(laplace_ext)


def spatial_gradient(field: Field,
                     boundary: Extrapolation = None,
                     at: str = 'center',
                     dims: math.DimFilter = spatial,
                     stack_dim: Union[Shape, str] = channel('vector'),
                     order=2,
                     implicit: Solve = None,
                     implicitness: int = None,
                     scheme=None,
                     upwind: Field = None,
                     gradient_extrapolation: Extrapolation = None):
    """
    Finite difference spatial_gradient.

    This function can operate in two modes:

    * `type=CenteredGrid` approximates the spatial_gradient at cell centers using central differences
    * `type=StaggeredGrid` computes the spatial_gradient at face centers of neighbouring cells

    Args:
        field: centered grid of any number of dimensions (scalar field, vector field, tensor field)
        boundary: Boundary conditions of the gradient field.
        at: Either `'face'` or `'center'`
        dims: Along which dimensions to compute the spatial gradient. Only supported when `type==CenteredGrid`.
        stack_dim: Dimension to be added. This dimension lists the spatial_gradient w.r.t. the spatial dimensions.
            The `field` must not have a dimension of the same name.
        order: Spatial order of accuracy.
            Higher orders entail larger stencils and more computation time but result in more accurate results assuming a large enough resolution.
            Supported: 2 explicit, 4 explicit, 6 implicit.
        implicit: When a `Solve` object is passed, performs an implicit operation with the specified solver and tolerances.
            Otherwise, an explicit stencil is used.
        implicitness: specifies the size of the implicit stencil in case an implicit treatment is used
        gradient_extrapolation: Alias for `boundary`.
        scheme: For unstructured meshes only. Currently only `'green-gauss'` is supported.
        upwind: For unstructured meshes only. Whether to use upwind interpolation.

    Returns:
        spatial_gradient field of type `type`.
    """

    assert at in ['face', 'center']
    stack_dim = auto(stack_dim)
    if gradient_extrapolation is not None:
        assert boundary is None, f"Cannot specify both boundary and gradient_extrapolation"
        boundary = gradient_extrapolation
    if boundary is None:
        boundary = field.extrapolation.spatial_gradient()
    if gradient_extrapolation is None:
        gradient_extrapolation = boundary
    if field.is_mesh and at == 'center':
        assert stack_dim not in field.shape, f"Gradient dimension is already part of field {field.shape}. Please use a different dimension"
        boundary = boundary or field.boundary.spatial_gradient()
        if scheme == 'green-gauss':
            return green_gauss_gradient(field, stack_dim=stack_dim, boundary=boundary, order=order, upwind=upwind)
        elif scheme == 'least-squares':
            return least_squares_gradient(field, stack_dim=stack_dim, boundary=boundary)
        raise NotImplementedError(scheme)

    if 'vector' in field.shape:
        assert stack_dim.name != 'vector', "`stack_dim=vector` is inadmissible if the input is a vector grid"
        if field == StaggeredGrid:
            assert at == 'faces', "for a `StaggeredGrid` input only `type == StaggeredGrid` is possible"

    if at == 'faces':
        assert stack_dim.name == 'vector', f"spatial_gradient with type=StaggeredGrid requires stack_dim.name == 'vector' but got '{stack_dim.name}'"

    if gradient_extrapolation is None:
        gradient_extrapolation = field.extrapolation.spatial_gradient()

    if implicitness is None:
        implicitness = 0 if implicit is None else 2
    elif implicitness != 0:
        assert implicit is not None, "for implicit treatment a `Solve` is required"

    grad_dims = field.shape.only(dims).names

    if stack_dim is None:
        assert len(grad_dims) == 1, "`stack_dim` `None` is only possible with single `grad_dim`"
    else:
        stack_dim = stack_dim.with_size(grad_dims)

    if order == 2:
        if at == 'center':
            values = math.spatial_gradient(field.values, field.dx.vector.as_channel(name=stack_dim.name),
                                           difference='central', padding=field.extrapolation, stack_dim=stack_dim)
            return CenteredGrid(values, bounds=field.bounds, extrapolation=gradient_extrapolation)
        elif at == 'face':
            assert stack_dim.name == 'vector'
            return stagger(field, lambda lower, upper: (upper - lower) / field.dx.vector.as_dual(), gradient_extrapolation)

    result_components = [
        perform_finite_difference_operation(field.values, dim, 1, field.dx.vector[dim], field.extrapolation,
                                            gradient_extrapolation, at, order, implicit, implicitness)
        for dim in field.shape.only(grad_dims).names]

    if at == 'center':
        result = field.with_values(math.stack(result_components, stack_dim))
        result = result.with_extrapolation(gradient_extrapolation)
    else:
        result = StaggeredGrid(
            math.stack(result_components, stack_dim.as_dual()),
            bounds=field.bounds, extrapolation=gradient_extrapolation)

    if at == 'center' and gradient_extrapolation == math.extrapolation.NONE:
        result = result.with_bounds(Box(field.bounds.lower - field.dx, field.bounds.upper + field.dx))
    else:
        result = result.with_bounds(field.bounds)

    return result


def get_stencils(order, differentiation_order, input_ext=None, implicit_order=0, border_stencil=True, staggered=False,
                 output_boundary_valid=False, left_border_one_sided=False):

    input_boundary_valid, bc_affin_lin, bc_deriv, bc_value = False, False, None, None
    if input_ext == extrapolation.ZERO_GRADIENT:
        bc_affin_lin = True
        bc_deriv = 1
        bc_value = 0
    elif isinstance(input_ext, extrapolation.ConstantExtrapolation):
        bc_affin_lin = True
        bc_deriv = 0
        bc_value = input_ext.value

    extend = int(math.ceil((order - implicit_order) / 2)) + int((differentiation_order - 1) / 2)
    rhs_extend = int(math.ceil(implicit_order / 2))

    shifts = [*range(-extend, extend + 1)]
    rhs_shifts = [*range(-rhs_extend, rhs_extend + 1)] if implicit_order else []

    v_ns_b0, rhs_v_ns_b0 = [], []
    max_extend = max(extend, rhs_extend)
    if border_stencil:
        for i in range(1, max_extend + 1):
            off = max(0, extend - max_extend + i)  # non defining boundary
            off_rhs = max(0, rhs_extend - max_extend + i + (not output_boundary_valid and staggered))
            n_shifts = [*range(-extend + off, extend + 1 + off + off_rhs + (differentiation_order % 2 == 0))]
            rhs_n_shifts = [*range(-rhs_extend + off_rhs, rhs_extend + 1)] if implicit_order else []

            if staggered:
                bc = None
                n_shifts = [n + 1 for n in n_shifts]
                coefficient_shifts = [n - 0.5 for n in n_shifts]
                if input_boundary_valid or bc_affin_lin:
                    del coefficient_shifts[-1]
                    del n_shifts[-1]
                    if input_boundary_valid:
                        coefficient_shifts.insert(0, coefficient_shifts[0] - 0.5)
                        n_shifts.insert(0, n_shifts[0] - 1)
                    else:
                        bc = coefficient_shifts[0] - 0.5, bc_deriv, bc_value

                n_values, n_values_rhs, n_affin_lin = get_coefficients(coefficient_shifts, differentiation_order,
                                                                       rhs_n_shifts, bc)
            else:
                bc = None
                coefficient_shifts = n_shifts.copy()
                if input_boundary_valid or bc_affin_lin:
                    del coefficient_shifts[-1]
                    del n_shifts[-1]
                    if input_boundary_valid:
                        coefficient_shifts.insert(0, coefficient_shifts[0] - 0.5)
                        n_shifts.insert(0, n_shifts[0] - 1)
                        bc = None
                    else:
                        bc = coefficient_shifts[0] - 0.5, bc_deriv, bc_value

                n_values, n_values_rhs, n_affin_lin = get_coefficients(coefficient_shifts, differentiation_order,
                                                                       rhs_n_shifts, bc)

            if left_border_one_sided:
                n_values = [v * (-1) ** differentiation_order for v in reversed(n_values)]
                if staggered:
                    n_shifts = [-s + 1 for s in reversed(n_shifts)]
                else:
                    n_shifts = [-s for s in reversed(n_shifts)]
                n_values_rhs = [v for v in reversed(n_values_rhs)]
                rhs_n_shifts = [-s for s in reversed(rhs_n_shifts)]

            v_ns_b0.insert(0, [n_values, n_shifts, [n_affin_lin]])
            rhs_v_ns_b0.insert(0, [n_values_rhs, rhs_n_shifts, [0]])

        if staggered and not output_boundary_valid:
            del v_ns_b0[0]
            del rhs_v_ns_b0[0]

            if len(v_ns_b0) == 0:
                v_ns_b0 = [[[], [], []]]

            if len(rhs_v_ns_b0) == 0:
                rhs_v_ns_b0 = [[[], [], []]]

    else:
        if staggered:
            del shifts[0]
            values, rhs_values, affin_lin = get_coefficients([s - 0.5 for s in shifts], differentiation_order,
                                                             rhs_shifts)
        else:
            values, rhs_values, affin_lin = get_coefficients(shifts, differentiation_order, rhs_shifts)

        return values, shifts, rhs_values, rhs_shifts, [affin_lin]

    return [v_ns_b0, rhs_v_ns_b0]


def perform_finite_difference_operation(field: Tensor, dim: str, differentiation_order: int, dx: float,
                                        ext: Extrapolation,
                                        output_ext: Extrapolation = None,
                                        at: str = 'center',
                                        order=2,
                                        implicit: Solve = None,
                                        implicitness: int = None):

    if output_ext is None:
        output_ext = ext

    if implicitness is None:
        implicitness = 0 if implicit is None else 2
    elif implicitness != 0:
        assert implicit is not None, "for implicit treatment a `Solve` is required"

    assert dim in field.shape.spatial.names, "given Tensor needs to have the indicated spatial dimension"

    leaf_exts = set()
    extrapolation.map(lambda e: extrapolation._NoExtrapolation(leaf_exts.add(e)), ext)

    def is_one_sided(ext):
        return (ext == extrapolation.ZERO_GRADIENT or isinstance(ext, extrapolation.ConstantExtrapolation)) and order > 2

    one_sided_exts = [ext for ext in leaf_exts if is_one_sided(ext)]

    with math.NUMPY:
        base_values, base_shifts, base_rhs_values, base_rhs_shifts, base_affin_lin = get_stencils(order,
                                                                                                  differentiation_order,
                                                                                                  implicit_order=implicitness,
                                                                                                  border_stencil=False,
                                                                                                  staggered=at == 'faces')

        if one_sided_exts != []:
            one_sided_stencils = \
                [
                    [
                        [get_stencils(order, differentiation_order, input_ext, implicit_order=implicitness, border_stencil=True,
                                      staggered=at == 'faces', output_boundary_valid=out_valid, left_border_one_sided=left)
                         for input_ext in one_sided_exts]
                     for out_valid in [False, True]]
                for left in [False, True]]
        else:
            one_sided_stencils = [[[[[[[]]], [[[]]]]]]]

    expl_one_sided_stencil_tensor = [[[l3[0] for l3 in l2] for l2 in l1] for l1 in one_sided_stencils]
    impl_one_sided_stencil_tensor = [[[l3[1] for l3 in l2] for l2 in l1] for l1 in one_sided_stencils]

    if at == 'center':
        standard_mask = CenteredGrid(0, resolution=field.shape.non_batch)  # ToDo ed is this okay with batch dimensions?
    else:
        standard_mask = CenteredGrid(0, resolution=field.shape.non_batch + spatial(
            **{dim: sum(output_ext.valid_outer_faces(dim)) - 1}))

    output_valid_ext = extrapolation.combine_sides(**{dim: tuple(
        extrapolation.ONE if valid_tuple else extrapolation.ZERO for valid_tuple in
        output_ext.valid_outer_faces(dim)) for dim in field.shape.spatial.names})
    output_valid_mask = standard_mask.with_extrapolation(output_valid_ext)

    ext_valid_masks = []
    for ex in one_sided_exts:
        mask_ext = extrapolation.map(lambda e: extrapolation.ONE if e == ex else extrapolation.ZERO, ext)
        ext_valid_masks.append(standard_mask.with_extrapolation(mask_ext))

    one_sided_ext = extrapolation.map(lambda e: extrapolation.ONE if is_one_sided(e) else extrapolation.ZERO, ext)
    one_sided_mask = standard_mask.with_extrapolation(one_sided_ext)

    result = apply_stencils(field, ext, output_ext, dx, base_values, base_shifts, at, dim,
                            masks=(ext_valid_masks, output_valid_mask, one_sided_mask),
                            stencil_tensors=expl_one_sided_stencil_tensor,
                            differencing_order=differentiation_order)

    if implicit:
        implicit.x0 = result
        result = solve_linear(apply_stencils, result, solve=implicit, field_extrapolation=ext,
                              gradient_extrapolation=output_ext,
                              field_dx=dx, base_koeff=base_rhs_values, base_shifts=base_rhs_shifts,
                              type=CenteredGrid, dim=dim,
                              masks=(ext_valid_masks, output_valid_mask, one_sided_mask),
                              stencil_tensors=impl_one_sided_stencil_tensor, differencing_order=0)

    return result


@jit_compile_linear(auxiliary_args="field_extrapolation, gradient_extrapolation, field_dx, base_koeff, base_shifts, "
                                   "type, dim, masks, stencil_tensors, differencing_order")
def apply_stencils(field, field_extrapolation, gradient_extrapolation, field_dx, base_koeff, base_shifts, at, dim,
                   masks=None, stencil_tensors=None, differencing_order=1):
    from itertools import product
    spatial_dims = field.shape.spatial.names

    def apply_stencil(values_, needed_shifts_, affin_lin_):
        needed_shifts_ = [int(i) for i in needed_shifts_]
        base_widths = (max(-min(needed_shifts_), 0), max(max(needed_shifts_), 0))

        std_widths = (0, 0)
        if at == 'center':
            if gradient_extrapolation == math.extrapolation.NONE:
                base_widths = (base_widths[0] + 1, base_widths[1] + 1)
                std_widths = (1, 1)
        elif at == 'face':
            base_widths = (base_widths[0], base_widths[1] - 1)
            border_valid = gradient_extrapolation.valid_outer_faces(dim)
            base_widths = (border_valid[0] + base_widths[0], border_valid[1] + base_widths[1])
        else:
            raise ValueError(at)

        padded_component = math.pad(field,
                                    {dim_: base_widths if dim_ == dim else std_widths for dim_ in spatial_dims},
                                    field_extrapolation)

        shifted_component = math.shift(padded_component, tuple(needed_shifts_), stack_dim=None, padding=None, dims=dim)
        result_component = (sum([value * shift for value, shift in
                                 zip(values_, shifted_component)]) + affin_lin_) / (field_dx ** differencing_order)

        return result_component

    result_component = apply_stencil(base_koeff, base_shifts, 0)
    if masks is not None and stencil_tensors is not None:
        ext_valid_masks, output_valid_mask, one_sided_mask = masks
        one_mask = one_sided_mask.with_values(1).with_extrapolation(extrapolation.ONE)
        for ext_valid, out_valid, left_side in product(range(len(ext_valid_masks)), [0, 1], [0, 1]):
            stencils = stencil_tensors[left_side][out_valid][ext_valid]
            ext_valid_mask_ = ext_valid_masks[ext_valid]
            output_valid_mask_ = output_valid_mask if out_valid else one_mask - output_valid_mask
            mask = ext_valid_mask_ * output_valid_mask_ * one_sided_mask
            isolation_mask = 0  # for obstacles we will have to tinker around here
            for i, stencils_i in enumerate(stencils):
                values_b0, needed_shifts_b0, affin_lin_b0 = stencils_i
                if len(values_b0) != 0 and len(values_b0) != 0:
                    one_sided_components = apply_stencil(values_b0, needed_shifts_b0, affin_lin_b0[0])
                    mask_ = shift(mask, ((i + 1) if left_side else -(i + 1),), dims=dim, stack_dim=None)[0].values - isolation_mask
                    isolation_mask = isolation_mask + mask_
                    result_component = math.where(mask_, one_sided_components, result_component)

    return result_component

def green_gauss_gradient(u: Field, boundary: Extrapolation, stack_dim: Shape = channel('vector'), order=2, upwind: Field = None) -> Field:
    """Computes the Green-Gauss gradient of a field at the centroids."""
    u = u.at_faces(boundary=NONE, order=order, upwind=upwind)
    normals = rename_dims(u.geometry.face_normals, 'vector', stack_dim)
    grad = u.geometry.integrate_surface(normals * u.values) / u.geometry.volume
    grad = slice_off_constant_faces(grad, u.geometry.boundary_elements, boundary)
    return Field(u.geometry, grad, boundary)


def least_squares_gradient(u: Field, boundary: Extrapolation, stack_dim: Shape = channel('vector')) -> Field:
    """Computes the least-squares gradient of a field at the centroids."""
    u_nb = u.mesh.pad_boundary(u.values, mode=u.boundary)
    du = (u.mesh.connectivity * u_nb - u.values)
    d = u.face_centers - u.center
    initial_guess = Field(u.geometry, math.zeros(stack_dim.with_size(u.geometry.vector.item_names)), boundary)
    @jit_compile_linear
    def du_from_grad(grad):
        return math.dot(grad.values, stack_dim, d, 'vector')
    raise NotImplementedError("least_squares_gradient not yet implemented")
    return math.solve_linear(du_from_grad, du, Solve(x0=initial_guess))  # not yet implemented for least-squares or sparse outputs


def shift(grid: Field, offsets: tuple, stack_dim: Optional[Shape] = channel('shift'), dims=spatial, pad=True):
    """
    Wraps :func:`math.shift` for CenteredGrid.

    Args:
      grid: CenteredGrid: 
      offsets: tuple: 
      stack_dim:  (Default value = 'shift')
    """
    if pad:
        padding = grid.extrapolation
        new_bounds = grid.bounds
    else:
        padding = None
        max_lower_shift = min(offsets) if min(offsets) < 0 else 0
        max_upper_shift = max(offsets) if max(offsets) > 0 else 0
        w_lower = math.wrap([max_lower_shift if dim in dims else 0 for dim in grid.shape.spatial.names])
        w_upper = math.wrap([max_upper_shift if dim in dims else 0 for dim in grid.shape.spatial.names])
        new_bounds = Box(grid.bounds.lower - w_lower * grid.dx, grid.bounds.upper - w_upper * grid.dx)
    data = math.shift(grid.values, offsets, dims=dims, padding=padding, stack_dim=stack_dim)
    return [create_similar_grid(grid, data[i], grid.extrapolation, new_bounds) for i in range(len(offsets))]


def stagger(field: Field,
            face_function: Callable,
            boundary: float or math.extrapolation.Extrapolation,
            at='face',
            dims=spatial):
    """
    Creates a new grid by evaluating `face_function` given two neighbouring cells.
    One layer of missing cells is inferred from the extrapolation.
    
    This method returns a Field of type `type` which must be either StaggeredGrid or CenteredGrid.
    When returning a StaggeredGrid, the new values are sampled at the faces of neighbouring cells.
    When returning a CenteredGrid, the new grid has the same resolution as `field`.

    Args:
        field: Grid
        face_function: function mapping (value1: Tensor, value2: Tensor) -> center_value: Tensor
        boundary: extrapolation mode of the returned grid. Has no effect on the values.
        at: Where the result should be sampled, one of 'face', 'center'
        dims: Which dimensions to stagger. Defaults to all spatial axes.

    Returns:
        Grid sampled either at centers or faces depending on `at`.
    """
    boundary = as_boundary(boundary, field.geometry)
    all_lower = []
    all_upper = []
    dims = field.shape.only(dims, reorder=True).names
    if at == 'face':
        for dim in dims:
            valid_lo, valid_up = boundary.valid_outer_faces(dim)
            if valid_lo and valid_up:
                width_lower, width_upper = {dim: (1, 0)}, {dim: (0, 1)}
            elif valid_lo and not valid_up:
                width_lower, width_upper = {dim: (1, -1)}, {dim: (0, 0)}
            elif not valid_lo and valid_up:
                width_lower, width_upper = {dim: (0, 0)}, {dim: (-1, 1)}
            else:
                width_lower, width_upper = {dim: (0, -1)}, {dim: (-1, 0)}
            comp = field.values.vector[dim] if 'vector' in channel(field) else field.values
            all_lower.append(math.pad(comp, width_lower, field.extrapolation, bounds=field.bounds))
            all_upper.append(math.pad(comp, width_upper, field.extrapolation, bounds=field.bounds))
        all_upper = math.stack(all_upper, dual(vector=dims))
        all_lower = math.stack(all_lower, dual(vector=dims))
        values = face_function(all_lower, all_upper)
        result = Field(field.geometry, values, boundary)
        assert result.resolution == field.resolution
        return result
    else:
        assert at == 'center', f"type must be 'face' or 'center' but got '{at}'"
        left, right = math.shift(field.values, (-1, 1), dims=dims, padding=field.extrapolation, stack_dim=channel('vector'))
        values = face_function(left, right)
        return CenteredGrid(values, boundary, field.bounds)


def divergence(field: Field, order=2, implicit: Solve = None, upwind: Field = None,
                     implicitness: int = None) -> CenteredGrid:
    """
    Computes the divergence of a grid using finite differences.

    This function can operate in two modes depending on the type of `field`:

    * `CenteredGrid` approximates the divergence at cell centers using central differences
    * `StaggeredGrid` exactly computes the divergence at cell centers

    Args:
        field: vector field as `CenteredGrid` or `StaggeredGrid`
        order: Spatial order of accuracy.
            Higher orders entail larger stencils and more computation time but result in more accurate results assuming a large enough resolution.
            Supported: 2 explicit, 4 explicit, 6 implicit.
        implicit: When a `Solve` object is passed, performs an implicit operation with the specified solver and tolerances.
            Otherwise, an explicit stencil is used.
        implicitness: specifies the size of the implicit stencil in case an implicit treatment is used
        upwind: For unstructured meshes only. Whether to use upwind interpolation.

    Returns:
        Divergence field as `CenteredGrid`
    """

    if field.is_mesh:
        field = field.at_faces(boundary=NONE, order=order, upwind=upwind)
        div = field.geometry.integrate_flux(field.values, divide_volume=True)
        return Field(field.geometry, div, field.boundary.spatial_gradient())
    if order == 2:
        if field.is_staggered:
            field = bake_extrapolation(field)
            components = []
            for dim in field.shape.spatial.names:
                div_dim = math.spatial_gradient(field.vector[dim].values, field.dx, 'forward', None, dims=dim,
                                                stack_dim=None)
                components.append(div_dim)
            data = math.sum(components, dim='0')
            return CenteredGrid(data, bounds=field.bounds, extrapolation=field.extrapolation.spatial_gradient())
        elif field.is_centered:
            left, right = shift(field, (-1, 1), stack_dim=batch('div_'))
            grad = (right - left) / (field.dx * 2)
            components = [grad.vector[i].div_[i] for i in grad.div_.item_names]
            result = sum(components)
            return result

    else:
        components = [
            spatial_gradient(f, dims=dim, at='center', order=order, implicit=implicit, implicitness=implicitness,
                             stack_dim="sum:b").sum[0] for f, dim in zip(field.vector, field.shape.only(spatial).names)]

    return sum(components)


def curl(field: Field, at='corner'):
    """
    Computes the finite-difference curl of the give 2D `StaggeredGrid`.

    Args:
        field: `Field`
        at: Either `center` or `face`.
    """
    assert 'vector' in field.shape, f"curl requires a vector field but got {field}"
    assert field.spatial_rank in (2, 3), "curl is only defined in 2 and 3 spatial dimensions."
    if field.is_grid and field.is_staggered and field.spatial_rank == 2 and at == 'corner':
        x, y = field.vector.item_names
        values = field.with_boundary(None).values
        vx = math.pad(values.vector.dual[x], {y: (1, 1)}, field.boundary[{'vector': y}])
        vy = math.pad(values.vector.dual[y], {x: (1, 1)}, field.boundary[{'vector': x}])
        vy_dx = math.spatial_gradient(vy, dims=x, dx=field.dx[x], padding=None, stack_dim=None, difference='forward')
        vx_dy = math.spatial_gradient(vx, dims=y, dx=field.dx[y], padding=None, stack_dim=None, difference='forward')
        curl_val = vy_dx - vx_dy
        corners = UniformGrid(field.resolution + 1, Box(field.bounds.lower - field.dx / 2, field.bounds.upper + field.dx / 2))
        return Field(corners, curl_val, field.boundary.spatial_gradient())
    elif field.is_grid and field.is_centered and field.spatial_rank == 2 and at == 'corner':
        x, y = field.vector.item_names
        values = pad(field, 1).values
        diag_basis = wrap([(1, 1), (1, -1)], channel(diag='pos,neg'), dual(vector=[x, y]))
        diag_comp = diag_basis @ values
        ll = diag_comp[{x: slice(-1), y: slice(-1), 'diag': 'neg'}]
        ul = diag_comp[{x: slice(-1), y: slice(1, None), 'diag': 'pos'}]
        lr = diag_comp[{x: slice(1, None), y: slice(-1), 'diag': 'pos'}]
        ur = diag_comp[{x: slice(1, None), y: slice(1, None), 'diag': 'neg'}]
        curl_val = ll - ul + lr - ur
        corners = UniformGrid(field.resolution + 1, Box(field.bounds.lower - field.dx / 2, field.bounds.upper + field.dx / 2))
        return Field(corners, curl_val, field.boundary.spatial_gradient())
    # if field.is_grid and not field.is_staggered and field.spatial_rank == 2:
    #     if 'vector' not in field.shape and at == 'face':
    #         # 2D curl of scalar field
    #         grad = math.spatial_gradient(field.values, dx=field.dx, difference='forward', padding=None, stack_dim=channel('vector'))
    #         result = grad.vector[::-1] * (1, -1)  # (d/dy, -d/dx)
    #         bounds = Box(field.bounds.lower + 0.5 * field.dx, field.bounds.upper - 0.5 * field.dx)  # lose 1 cell per dimension
    #         return StaggeredGrid(result, bounds=bounds, extrapolation=field.extrapolation.spatial_gradient())
    #     if 'vector' in field.shape and at == 'center':
    #         # 2D curl of vector field
    #         x, y = field.resolution.names
    #         vy_dx = math.spatial_gradient(field.values.vector[1], dx=field.dx[0], padding=field.extrapolation, dims=x, stack_dim=None)
    #         vx_dy = math.spatial_gradient(field.values.vector[0], dx=field.dx[1], padding=field.extrapolation, dims=y, stack_dim=None)
    #         c = vy_dx - vx_dy
    #         return field.with_values(c)
    # elif field.is_grid and field.is_staggered and field.spatial_rank == 2:
    #     if at == 'center':
    #         values = bake_extrapolation(field).values
    #         x_padded = math.pad(values.vector['x'], {'y': (1, 1)}, field.extrapolation[{'vector': 'x'}], bounds=field.bounds)
    #         y_padded = math.pad(values.vector['y'], {'x': (1, 1)}, field.extrapolation[{'vector': 'y'}], bounds=field.bounds)
    #         vx_dy = math.spatial_gradient(x_padded, field.dx, 'forward', None, dims='y', stack_dim=None)
    #         vy_dx = math.spatial_gradient(y_padded, field.dx, 'forward', None, dims='x', stack_dim=None)
    #         result = vy_dx - vx_dy
    #         return CenteredGrid(result, field.extrapolation.spatial_gradient(), field.bounds)
    elif field.is_grid and field.is_staggered and field.spatial_rank == 3 and at == 'corner':
        x, y, z = field.vector.item_names
        values = field.with_boundary(None).values
        vx = math.pad(values.vector.dual[x], {y: (1, 1), z: (1, 1)}, field.boundary[{'vector': y}])
        vy = math.pad(values.vector.dual[y], {x: (1, 1), z: (1, 1)}, field.boundary[{'vector': x}])
        vz = math.pad(values.vector.dual[z], {x: (1, 1), y: (1, 1)}, field.boundary[{'vector': z}])
        vx_dy = neighbor_mean(math.spatial_gradient(vx, dims=y, dx=field.dx[y], padding=None, stack_dim=None, difference='forward'), z)
        vx_dz = neighbor_mean(math.spatial_gradient(vx, dims=z, dx=field.dx[z], padding=None, stack_dim=None, difference='forward'), y)
        vy_dx = neighbor_mean(math.spatial_gradient(vy, dims=x, dx=field.dx[x], padding=None, stack_dim=None, difference='forward'), z)
        vy_dz = neighbor_mean(math.spatial_gradient(vy, dims=z, dx=field.dx[z], padding=None, stack_dim=None, difference='forward'), x)
        vz_dx = neighbor_mean(math.spatial_gradient(vz, dims=x, dx=field.dx[x], padding=None, stack_dim=None, difference='forward'), y)
        vz_dy = neighbor_mean(math.spatial_gradient(vz, dims=y, dx=field.dx[y], padding=None, stack_dim=None, difference='forward'), x)
        curl_val = math.stack([vz_dy-vy_dz, vx_dz-vz_dx, vy_dx-vx_dy], field.shape['vector'])
        corners = UniformGrid(field.resolution + 1, Box(field.bounds.lower - field.dx / 2, field.bounds.upper + field.dx / 2))
        return Field(corners, curl_val, field.boundary.spatial_gradient())
    elif field.is_grid and field.is_centered and field.spatial_rank == 3 and at == 'corner':
        raise NotImplementedError
        x, y, z = field.vector.item_names
        values = pad(field, 1).values
        # ToDo 8 diag offset vectors, account for cell stretching
        # Then sum (offset x v) / |offset|^2 ??
        diag_basis = wrap([(1, 1), (1, -1)], channel(diag='pos,neg'), dual(vector=[x, y]))
        diag_comp = diag_basis @ values
        ll = diag_comp[{x: slice(-1), y: slice(-1), 'diag': 'neg'}]
        ul = diag_comp[{x: slice(-1), y: slice(1, None), 'diag': 'pos'}]
        lr = diag_comp[{x: slice(1, None), y: slice(-1), 'diag': 'pos'}]
        ur = diag_comp[{x: slice(1, None), y: slice(1, None), 'diag': 'neg'}]
        curl_val = ll - ul + lr - ur
        corners = UniformGrid(field.resolution + 1, Box(field.bounds.lower - field.dx / 2, field.bounds.upper + field.dx / 2))
        return Field(corners, curl_val, field.boundary.spatial_gradient())
    raise NotImplementedError("Only 2D curl at corner currently supported")


def fourier_laplace(grid: Field, times=1) -> Field:
    """ See `phi.math.fourier_laplace()` """
    assert grid.extrapolation.spatial_gradient() == math.extrapolation.PERIODIC
    values = math.fourier_laplace(grid.values, dx=grid.dx, times=times)
    return type(grid)(values=values, bounds=grid.bounds, extrapolation=grid.extrapolation)


def fourier_poisson(grid: Field, times=1) -> Field:
    """ See `phi.math.fourier_poisson()` """
    assert grid.extrapolation.spatial_gradient() == math.extrapolation.PERIODIC
    values = math.fourier_poisson(grid.values, dx=grid.dx, times=times)
    return type(grid)(values=values, bounds=grid.bounds, extrapolation=grid.extrapolation)


def native_call(f, *inputs, channels_last=None, channel_dim='vector', extrapolation=None) -> Union[Field, Tensor]:
    """
    Similar to `phi.math.native_call()`.

    Args:
        f: Function to be called on native tensors of `inputs.values`.
            The function output must have the same dimension layout as the inputs and the batch size must be identical.
        *inputs: `Field` or `phi.Tensor` instances.
        extrapolation: (Optional) Extrapolation of the output field. If `None`, uses the extrapolation of the first input field.

    Returns:
        `Field` matching the first `Field` in `inputs`.
    """
    input_tensors = [i.uniform_values() if isinstance(i, Field) else tensor(i) for i in inputs]
    values = math.native_call(f, *input_tensors, channels_last=channels_last, channel_dim=channel_dim)
    for i in inputs:
        if isinstance(i, Field):
            if not i.values.shape.is_uniform:
                values = unstack_staggered_tensor(values, i.boundary)
            result = i.with_values(values=values)
            if extrapolation is not None:
                result = result.with_extrapolation(extrapolation)
            return result
    else:
        raise AssertionError("At least one input must be a Field.")


def data_bounds(loc: Union[Field, Tensor]) -> Box:
    if isinstance(loc, Field):
        loc = loc.points
    assert isinstance(loc, Tensor), f"loc must be a Tensor or Field but got {type(loc)}"
    min_vec = math.min(loc, dim=loc.shape.non_batch.non_channel)
    max_vec = math.max(loc, dim=loc.shape.non_batch.non_channel)
    return Box(min_vec, max_vec)


def mean(field: Field, dim=lambda s: s.non_channel.non_batch) -> Tensor:
    """
    Computes the mean value by reducing all spatial / instance dimensions.

    Args:
        field: `Field`

    Returns:
        `phi.Tensor`
    """
    if field.is_grid:
        result = math.mean(field.values, dim=dim)
    else:
        result = math.mean(field.values, dim=dim, weight=field.geometry.volume)
    if (instance(field.geometry) & spatial(field.geometry)) in result.shape:
        return field.with_values(result)
    return result


def normalize(field: Field, norm: Field, epsilon=1e-5):
    """ Multiplies the values of `field` so that its sum matches the source. """
    data = math.normalize_to(field.values, norm.values, epsilon)
    return field.with_values(data)


def center_of_mass(density: Field):
    """
    Compute the center of mass of a density field.

    Args:
        density: Scalar `Field`

    Returns:
        `Tensor` holding only batch dimensions.
    """
    assert 'vector' not in density.shape
    return mean(density.points * density) / mean(density)


def pad(grid: Field, widths: Union[int, tuple, list, dict]) -> Field:
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
    widths_list = [widths[axis] if axis in widths.keys() else (0, 0) for axis in grid.shape.spatial.names]
    if grid.is_grid:
        if grid.is_staggered:
            data = math.pad(grid.values.vector.dual.as_channel(), widths, grid.extrapolation, bounds=grid.bounds).vector.as_dual()
        else:
            data = math.pad(grid.values, widths, grid.extrapolation, bounds=grid.bounds)
        w_lower = math.wrap([w[0] for w in widths_list])
        w_upper = math.wrap([w[1] for w in widths_list])
        bounds = Box(grid.bounds.lower - w_lower * grid.dx, grid.bounds.upper + w_upper * grid.dx)
        return create_similar_grid(grid, data, grid.extrapolation, bounds)
    raise NotImplementedError(f"{type(grid)} not supported. Only Grid instances allowed.")


def create_similar_grid(grid: Field, data, extrapolation, bounds):
    if grid.is_grid and not grid.is_staggered:
        return CenteredGrid(data, extrapolation, bounds)
    elif grid.is_grid and grid.is_staggered:
        return StaggeredGrid(data, extrapolation, bounds)
    else:
        raise ValueError(grid)


def downsample2x(grid: Field) -> Field:
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
    if grid.is_grid and grid.is_centered:
        values = math.downsample2x(grid.values, grid.extrapolation)
        return CenteredGrid(values, bounds=grid.bounds, extrapolation=grid.extrapolation)
    elif grid.is_grid and grid.is_staggered:
        grid_ = grid.with_boundary(extrapolation.NONE)
        values = {}
        for dim in grid.vector.item_names:
            odd_discarded = grid_.values[{'~vector': dim, dim: slice(None, None, 2)}]
            others_interpolated = math.downsample2x(odd_discarded, grid.extrapolation, dims=grid.shape.spatial.without(dim))
            values[dim] = others_interpolated
        return StaggeredGrid(math.stack(values, dual('vector')), None, grid.bounds).with_extrapolation(grid.extrapolation)
    else:
        raise ValueError(grid)


def upsample2x(grid: Field) -> Field:
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
    assert grid.is_grid, f"upsample2x only supported for grids but got {grid}"
    if grid.is_centered:
        values = math.upsample2x(grid.values, grid.extrapolation)
        return CenteredGrid(values, bounds=grid.bounds, extrapolation=grid.extrapolation)
    elif grid.is_staggered:
        raise NotImplementedError()
    else:
        raise ValueError(type(grid))


def concat(fields: Sequence[Field], dim: str or Shape) -> Field:
    """
    Concatenates the given `Field`s along `dim`.

    See Also:
        `stack()`.

    Args:
        fields: List of matching `Field` instances.
        dim: Concatenation dimension as `Shape`. Size is ignored.

    Returns:
        `Field` matching concatenated fields.
    """
    assert all(isinstance(f, Field) for f in fields)
    assert all(isinstance(f, type(fields[0])) for f in fields)
    if any(f.extrapolation != fields[0].extrapolation for f in fields):
        raise NotImplementedError("Concatenating extrapolations not supported")
    if fields[0].is_grid:
        values = math.concat([f.values for f in fields], dim)
        return fields[0].with_values(values)
    elif fields[0].is_point_cloud or fields[0].is_graph:
        geometry = geom.concat([f.geometry for f in fields], dim)
        values = math.concat([math.expand(f.values, f.shape.only(dim)) for f in fields], dim)
        return PointCloud(elements=geometry, values=values, extrapolation=fields[0].extrapolation)
    elif fields[0].is_mesh:
        assert all([f.geometry == fields[0].geometry for f in fields])
        values = math.concat([math.expand(f.values, f.shape.only(dim)) for f in fields], dim)
        return Field(fields[0].geometry, values, fields[0].extrapolation)
    raise NotImplementedError(type(fields[0]))


def stack(fields: Sequence[Field], dim: Shape, dim_bounds: Box = None):
    """
    Stacks the given `Field`s along `dim`.

    See Also:
        `concat()`.

    Args:
        fields: List of matching `Field` instances.
        dim: Stack dimension as `Shape`. Size is ignored.
        dim_bounds: `Box` defining the physical size for `dim`.

    Returns:
        `Field` matching stacked fields.
    """
    assert all(isinstance(f, Field) for f in fields), f"All fields must be Fields of the same type but got {fields}"
    assert all(isinstance(f, type(fields[0])) for f in fields), f"All fields must be Fields of the same type but got {fields}"
    if any([f.sampled_at != fields[0].sampled_at for f in fields]):
        return math.layout(fields, dim)
    if any(f.boundary != fields[0].boundary for f in fields):
        boundary = math.stack([f.boundary for f in fields], dim)
    else:
        boundary = fields[0].boundary
    if fields[0].is_grid:
        values = math.stack([f.values for f in fields], dim)
        if spatial(dim):
            if dim_bounds is None:
                dim_bounds = Box(**{dim.name: len(fields)})
            return grid(values, boundary, fields[0].bounds * dim_bounds)
        else:
            return fields[0].with_values(values).with_boundary(boundary)
    else:
        values = math.stack([f.values for f in fields], dim)
        geometry = fields[0].geometry if all(f.geometry == fields[0].geometry for f in fields) else math.stack([f.geometry for f in fields], dim, layout_non_matching=True)
        if isinstance(geometry, Tensor):
            from phi.geom._geom_ops import GeometryStack
            geometry = GeometryStack(geometry)
        return Field(geometry, values, boundary)


def assert_close(*fields: Field or Tensor or Number,
                 rel_tolerance: float = 1e-5,
                 abs_tolerance: float = 0,
                 msg: str = "",
                 verbose: bool = True):
    """ Raises an AssertionError if the `values` of the given fields are not close. See `phi.math.assert_close()`. """
    f0 = next(filter(lambda t: isinstance(t, Field), fields))
    values = [(f @ f0).values if isinstance(f, Field) else math.wrap(f) for f in fields]
    math.assert_close(*values, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance, msg=msg, verbose=verbose)


def where(mask: Field or Geometry or float, field_true: Field or float, field_false: Field or float) -> Field:
    """
    Element-wise where operation.
    Picks the value of `field_true` where `mask=1 / True` and the value of `field_false` where `mask=0 / False`.

    The fields are automatically resampled if necessary, preferring the sample points of `mask`.
    At least one of the arguments must be a `Field`.

    Args:
        mask: `Field` or `Geometry` object.
        field_true: `Field`
        field_false: `Field`

    Returns:
        `Field`
    """
    field_true, field_false, mask = _auto_resample(field_true, field_false, mask)
    values = math.where(mask.values, field_true.values, field_false.values)
    return field_true.with_values(values)


def maximum(f1: Field or Geometry or float, f2: Field or Geometry or float):
    """
    Element-wise maximum.
    One of the given fields needs to be an instance of `Field` and the the result will be sampled at the corresponding points.
    If both are `Fields` but have different points, `f1` takes priority.

    Args:
        f1: `Field` or `Geometry` or constant.
        f2: `Field` or `Geometry` or constant.

    Returns:
        `Field`
    """
    f1, f2 = _auto_resample(f1, f2)
    return f1.with_values(math.maximum(f1.values, f2.values))


def minimum(f1: Field or Geometry or float, f2: Field or Geometry or float):
    """
    Element-wise minimum.
    One of the given fields needs to be an instance of `Field` and the the result will be sampled at the corresponding points.
    If both are `Fields` but have different points, `f1` takes priority.

    Args:
        f1: `Field` or `Geometry` or constant.
        f2: `Field` or `Geometry` or constant.

    Returns:
        `Field`
    """
    f1, f2 = _auto_resample(f1, f2)
    return f1.with_values(math.minimum(f1.values, f2.values))


def _auto_resample(*fields: Field):
    """ Prefers extrapolation from first Field """
    from ._resample import resample
    for sampled_field in fields:
        if isinstance(sampled_field, Field):
            return [resample(f, sampled_field) for f in fields]
    raise AssertionError(f"At least one argument must be a Field but got {fields}")


def vec_length(field: Field):
    """ See `phi.math.vec_abs()` """
    assert isinstance(field, Field), f"Field required but got {type(field).__name__}"
    if field.is_grid and field.is_staggered:
        field = field.at_centers()
    return field.with_values(math.vec_abs(field.values))


def vec_squared(field: Field):
    """ See `phi.math.vec_squared()` """
    if field.is_grid and field.is_staggered:
        field = field.at_centers()
    return field.with_values(math.vec_squared(field.values))


def finite_fill(grid: Field, distance=1, diagonal=True) -> Field:
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
    if grid.is_grid and grid.is_centered:
        new_values = math.finite_fill(grid.values, distance=distance, diagonal=diagonal, padding=grid.extrapolation)
        return grid.with_values(new_values)
    elif grid.is_grid and grid.is_staggered:
        new_values = [finite_fill(c, distance=distance, diagonal=diagonal).values for c in grid.vector]
        return grid.with_values(math.stack(new_values, channel(grid).as_dual()))
    else:
        raise ValueError(grid)


def discretize(grid: Field, filled_fraction=0.25):
    """ Treats channel dimensions as batch dimensions. """
    import numpy as np
    data = math.reshaped_native(grid.values, [grid.shape.non_spatial, grid.shape.spatial])
    ranked_idx = np.argsort(data, axis=-1)
    filled_idx = ranked_idx[:, int(round(grid.shape.spatial.volume * (1 - filled_fraction))):]
    filled = np.zeros_like(data)
    np.put_along_axis(filled, filled_idx, 1, axis=-1)
    filled_t = math.reshaped_tensor(filled, [grid.shape.non_spatial, grid.shape.spatial])
    return grid.with_values(filled_t)


def integrate(field: Field, region: Geometry, **kwargs) -> Tensor:
    """
    Computes *∫<sub>R</sub> f(x) dx<sup>d</sup>* , where *f* denotes the `Field`, *R* the `region` and *d* the number of spatial dimensions (`d=field.shape.spatial_rank`).
    Depending on the `sample` implementation for `field`, the integral may be a rough approximation.

    This method is currently only implemented for `CenteredGrid`.

    Args:
        field: `Field` to integrate.
        region: Region to integrate over.
        **kwargs: Specify numerical scheme.

    Returns:
        Integral as `phi.Tensor`
    """
    if not field.is_grid and not field.is_staggered:
        raise NotImplementedError()
    return sample(field, region, **kwargs) * region.volume


def pack_dims(field: Field,
              dims: Shape or tuple or list or str,
              packed_dim: Shape,
              pos: int or None = None) -> Field:
    """
    Currently only supports grids and non-spatial dimensions.

    See Also:
        `phi.math.pack_dims()`.

    Args:
        field: `Field`

    Returns:
        `Field` of same type as `field`.
    """
    if isinstance(field, Field):
        if spatial(field.shape.only(dims)):
            raise NotImplementedError("Packing spatial dimensions not supported for grids")
        return field.with_values(math.pack_dims(field.values, dims, packed_dim, pos))
    else:
        raise NotImplementedError()


def support(field: Field, list_dim: Shape or str = instance('nonzero')) -> Tensor:
    """
    Returns the points at which the field values are non-zero.

    Args:
        field: `Field`
        list_dim: Dimension to list the non-zero values.

    Returns:
        `Tensor` with shape `(list_dim, vector)`
    """
    return field.points[math.nonzero(field.values, list_dim=list_dim)]


def mask(obj: Field or Geometry) -> Field:
    """
    Returns a `Field` that masks the inside (or non-zero values when `obj` is a grid) of a physical object.
    The mask takes the value 1 inside the object and 0 outside.
    For `CenteredGrid` and `StaggeredGrid`, the mask labels non-zero non-NaN entries as 1 and all other values as 0

    Returns:
        `Grid` type or `PointCloud`
    """
    if isinstance(obj, Geometry):
        return Field(obj, 1, 0)
    assert isinstance(obj, Field), f"obj must be a Geometry or Field but got {type(obj)}"
    if obj.is_grid and not obj.is_staggered:
        values = math.cast(obj.values != 0, int)
        return obj.with_values(values)
    elif obj.is_staggered:
        raise NotImplementedError
    else:
        return Field(obj.elements, 1, math.extrapolation.remove_constant_offset(obj.extrapolation))


def get_coefficients(offsets, derivative, lhs_offsets=[], boundary_condition=None):

        def taylor_coeff(offset, n, deriv):
            coeff = (offset) ** abs(n - deriv) / factorial(n - deriv)
            res = math.where(n - deriv >= 0, coeff, 0)
            return res

        handle_zero = 0 in lhs_offsets
        if handle_zero:
            lhs_offsets = lhs_offsets.copy()
            zero_index = lhs_offsets.index(0)
            lhs_offsets.remove(0)

        bc = boundary_condition is not None
        if bc:
            bc_offset, bc_deriv, bc_value = boundary_condition

        node_number = len(offsets + lhs_offsets) + bc

        one = math.concat([zeros(channel(x=derivative)), ones(channel(x=1)), zeros(channel(x=node_number - derivative - 1))],
                     'x')
        # ToDo ed switch zero(...) -> phiml.math.expand(0, ...)

        arange = tensor(np.arange(node_number), channel('x'))
        coeff = taylor_coeff(tensor(offsets, dual('x')), arange, 0)
        coeff_lhs = taylor_coeff(tensor(lhs_offsets, dual('x')), arange, derivative)
        mat = math.concat([coeff, coeff_lhs], '~x')

        if bc:
            coeff_bc = taylor_coeff(tensor([bc_offset], dual('x')), arange, bc_deriv)
            mat = math.concat([mat, coeff_bc], '~x')

        np_mat = mat.numpy('x, ~x')
        np_b = one.numpy('x')
        coeff = np.linalg.solve(np_mat, np_b)
        ret = list(coeff)
        values, lhs_values = ret[:len(offsets)], ret[len(offsets):len(offsets + lhs_offsets)]
        lhs_values = [-v for v in lhs_values]

        bc_offset = 0
        if bc:
            bc_offset = ret[-1] * bc_value

        if handle_zero:
            lhs_values.insert(zero_index, 1)

        return values, lhs_values, bc_offset

# def connect(obj: Field, connections: Tensor) -> Mesh:
#     """
#     Build a `Mesh` by connecting elements from a field.
#
#     See Also:
#         `connect_neighbors()`.
#
#     Args:
#         obj: `PointCloud` or `Mesh`.
#         connections: Connectivity matrix. Any non-zero entry represents a connection.
#
#     Returns:
#         `Mesh`
#     """
#     if isinstance(obj, (PointCloud, Mesh)):
#         return Mesh(obj.elements, connections, obj.values, extrapolation=obj.extrapolation, bounds=obj.bounds)
#     else:
#         raise ValueError(f"connect requires a PointCloud or Mesh but got {type(obj)}")
#
#
# def connect_neighbors(obj: Field, max_distance: float or Tensor, format: str = 'dense') -> Mesh:
#     """
#     Build  a `Mesh` by connecting proximate elements of a `Field`.
#
#     See Also:
#         `connect()`.
#
#     Args:
#         obj: `PointCloud`, `Mesh`, `CenteredGrid` or `StaggeredGrid`.
#         max_distance: Connectivity threshold distance. Elements further apart than this will not be connected.
#         format: Connectivity matrix format, `'dense'`, `'coo'` or `'csr'`.
#
#     Returns:
#         `Mesh`.
#     """
#     if isinstance(obj, CenteredGrid):
#         elements = flatten(obj.elements, instance('elements'))
#         values = math.pack_dims(obj.values, spatial, instance('elements'))
#         obj = PointCloud(elements, values, obj.extrapolation, bounds=obj.bounds)
#     elif isinstance(obj, StaggeredGrid):
#         elements = flatten(obj.elements, instance('elements'), flatten_batch=True)
#         values = math.pack_dims(obj.values, spatial(obj.values).names + ('vector',), instance('elements'))
#         obj = PointCloud(elements, values, obj.extrapolation, bounds=obj.bounds)
#     assert isinstance(obj, (PointCloud, Mesh)), f"obj must be a PointCloud, Mesh or Grid but got {type(obj)}"
#     points = math.rename_dims(obj.elements, spatial, instance).center
#     dx = math.pairwise_distances(points, max_distance=max_distance, format=format)
#     con = math.vec_length(dx) > 0
#     return connect(obj, con)

def safe_mul(x, y):
    """See `phiml.math.safe_mul()`"""
    x, y = _auto_resample(x, y)
    values = math.safe_mul(x.values, y.values)
    return x.with_values(values)
