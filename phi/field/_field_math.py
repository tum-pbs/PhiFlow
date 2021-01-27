import warnings
from functools import wraps
from typing import TypeVar

import numpy as np
from phi import math
from phi.geom import Box, Geometry
from . import StaggeredGrid, ConstantField, HardGeometryMask
from ._field import Field, SampledField
from ._grid import CenteredGrid, Grid
from ..math import tensor


def laplace(field: Grid, axes=None):
    result = field._op1(lambda tensor: math.laplace(tensor, dx=field.dx, padding=field.extrapolation, dims=axes))
    return result


def gradient(field: CenteredGrid, type: type = CenteredGrid, stack_dim='vector'):
    """
    Finite difference gradient.

    This function can operate in two modes:

    * `type=CenteredGrid` approximates the gradient at cell centers using central differences
    * `type=StaggeredGrid` computes the gradient at face centers of neighbouring cells

    Args:
        field: centered grid of any number of dimensions (scalar field, vector field, tensor field)
        type: either `CenteredGrid` or `StaggeredGrid`
        stack_dim: name of dimension to be added. This dimension lists the gradient w.r.t. the spatial dimensions.
            The `field` must not have a dimension of the same name.

    Returns:
        gradient field of type `type`.

    """
    if type == CenteredGrid:
        values = math.gradient(field.values, field.dx.vector.as_channel(name=stack_dim), difference='central', padding=field.extrapolation, stack_dim=stack_dim)
        return CenteredGrid(values, field.bounds, field.extrapolation.gradient())
    elif type == StaggeredGrid:
        assert stack_dim == 'vector'
        return stagger(field, lambda lower, upper: (upper - lower) / field.dx, field.extrapolation.gradient())
    raise NotImplementedError(f"{type(field)} not supported. Only CenteredGrid and StaggeredGrid allowed.")


def shift(grid: CenteredGrid, offsets: tuple, stack_dim='shift'):
    """
    Wraps :func:`math.shift` for CenteredGrid.

    Args:
      grid: CenteredGrid: 
      offsets: tuple: 
      stack_dim:  (Default value = 'shift')

    Returns:

    """
    data = math.shift(grid.values, offsets, padding=grid.extrapolation, stack_dim=stack_dim)
    return [CenteredGrid(data[i], grid.box, grid.extrapolation) for i in range(len(offsets))]


def stagger(field: CenteredGrid, face_function: callable, extrapolation: math.extrapolation.Extrapolation, type: type = StaggeredGrid):
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
      face_function: callable: 
      extrapolation: math.extrapolation.Extrapolation: 
      type: type:  (Default value = StaggeredGrid)

    Returns:
      grid of type matching the `type` argument

    """
    all_lower = []
    all_upper = []
    if type == StaggeredGrid:
        for dim in field.shape.spatial.names:
            all_upper.append(math.pad(field.values, {dim: (0, 1)}, field.extrapolation))
            all_lower.append(math.pad(field.values, {dim: (1, 0)}, field.extrapolation))
        all_upper = math.channel_stack(all_upper, 'vector')
        all_lower = math.channel_stack(all_lower, 'vector')
        values = face_function(all_lower, all_upper)
        return StaggeredGrid(values, field.bounds, extrapolation)
    elif type == CenteredGrid:
        left, right = math.shift(field.values, (-1, 1), padding=field.extrapolation, stack_dim='vector')
        values = face_function(left, right)
        return CenteredGrid(values, field.bounds, extrapolation)
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
        components = []
        for i, dim in enumerate(field.shape.spatial.names):
            div_dim = math.gradient(field.values.vector[i], dx=field.dx[i], difference='forward', padding=None, dims=[dim]).gradient[0]
            components.append(div_dim)
        data = math.sum(components, 0)
        return CenteredGrid(data, field.box, field.extrapolation.gradient())
    elif isinstance(field, CenteredGrid):
        left, right = shift(field, (-1, 1), stack_dim='div_')
        grad = (right - left) / (field.dx * 2)
        components = [grad.vector[i].div_[i] for i in range(grad.div_.size)]
        result = sum(components)
        return result
    else:
        raise NotImplementedError(f"{type(field)} not supported. Only StaggeredGrid allowed.")


FieldType = TypeVar('FieldType', bound=Field)
GridType = TypeVar('GridType', bound=Grid)


def diffuse(field: FieldType, diffusivity, dt, substeps=1) -> FieldType:
    """
    Simulate a finite-time diffusion process of the form dF/dt = α · ΔF on a given `Field` FieldType with diffusion coefficient α.
    
    If `field` is periodic (set via `extrapolation='periodic'`), diffusion may be simulated in Fourier space.
    Otherwise, finite differencing is used to approximate the

    Args:
      field: CenteredGrid, StaggeredGrid or ConstantField
      diffusivity: diffusion amount = diffusivity * dt
      dt: diffusion amount = diffusivity * dt
      substeps: number of iterations to use (Default value = 1)
      field: FieldType: 

    Returns:
      Field: Field of same type as `field`

    """
    if isinstance(field, ConstantField):
        return field
    assert isinstance(field, Grid), "Cannot diffuse field of type '%s'" % type(field)
    amount = diffusivity * dt
    if field.extrapolation == 'periodic' and not isinstance(amount, Field):
        fft_laplace = -(2 * np.pi) ** 2 * squared(fftfreq(field))
        diffuse_kernel = math.exp(fft_laplace * amount)
        return real(ifft(fft(field) * diffuse_kernel))
    else:
        if isinstance(amount, Field):
            amount = amount.at(field)
        for i in range(substeps):
            field += amount / substeps * laplace(field)
        return field


def solve(function, y: Grid, x0: Grid, solve_params: math.Solve, callback=None):
    if callback is not None:
        def field_callback(x):
            x = x0._with(x)
            callback(x)
    else:
        field_callback = None

    data_function = expose_tensors(function, x0)
    converged, x, iterations = math.solve(data_function, y.values, x0.values, solve_params, field_callback)
    return converged, x0._with(x), iterations


def expose_tensors(field_function, *proto_fields):
    @wraps(field_function)
    def wrapper(*field_data):
        fields = [proto._with(data) for data, proto in zip(field_data, proto_fields)]
        result = field_function(*fields)
        assert isinstance(result, SampledField), f"function must return an instance of SampledField but returned {result}"
        return result.values
    return wrapper


def data_bounds(field: SampledField):
    data = field.points
    min_vec = math.min(data, dim=data.shape.spatial.names)
    max_vec = math.max(data, dim=data.shape.spatial.names)
    return Box(min_vec, max_vec)


def mean(field: Grid):
    return math.mean(field.values, field.shape.spatial)


def normalize(field: SampledField, norm: SampledField, epsilon=1e-5):
    data = math.normalize_to(field.values, norm.values, epsilon)
    return field._with(data)


def pad(grid: Grid, widths: int or tuple or list or dict):
    if isinstance(widths, int):
        widths = {axis: (widths, widths) for axis in grid.shape.spatial.names}
    elif isinstance(widths, (tuple, list)):
        widths = {axis: (width if isinstance(width, (tuple, list)) else (width, width)) for axis, width in zip(grid.shape.spatial.names, widths)}
    else:
        assert isinstance(widths, dict)
    widths_list = [widths[axis] for axis in grid.shape.spatial.names]
    if isinstance(grid, Grid):
        data = math.pad(grid.values, widths, grid.extrapolation)
        w_lower = tensor([w[0] for w in widths_list])
        w_upper = tensor([w[1] for w in widths_list])
        box = Box(grid.box.lower - w_lower * grid.dx, grid.box.upper + w_upper * grid.dx)
        return type(grid)(data, box, grid.extrapolation)
    raise NotImplementedError(f"{type(grid)} not supported. Only Grid instances allowed.")


def divergence_free(vector_field: Grid, solve_params: math.LinearSolve = math.LinearSolve(None, 1e-5)):
    """
    Returns the divergence-free part of the given vector field.
    The boundary conditions are taken from `vector_field`.
    
    This function solves for a scalar potential with an iterative solver.

    Args:
      vector_field: vector grid
      solve_params: return: divergence-free vector field, scalar potential, number of iterations performed, divergence
      vector_field: Grid: 
      solve_params: math.LinearSolve:  (Default value = math.LinearSolve(None)
      1e-5): 

    Returns:
      divergence-free vector field, scalar potential, number of iterations performed, divergence

    """
    div = divergence(vector_field)
    div -= mean(div)
    pressure_extrapolation = vector_field.extrapolation  # periodic -> periodic, closed -> boundary, open -> zero
    pressure_guess = CenteredGrid.sample(0, vector_field.resolution, vector_field.box, extrapolation=pressure_extrapolation)
    converged, potential, iterations = solve(laplace, div, pressure_guess, solve_params)
    gradp = gradient(potential, type=StaggeredGrid)
    vector_field -= gradp
    return vector_field, potential, iterations, div


def squared(field: Field):
    raise NotImplementedError()


def real(field: Field):
    raise NotImplementedError()


def imag(field: Field):
    raise NotImplementedError()


def fftfreq(grid: Grid):
    raise NotImplementedError()


def fft(grid: Grid):
    raise NotImplementedError()


def ifft(grid: Grid):
    raise NotImplementedError()


def staggered_curl_2d(grid, pad_width=(1, 2)):
    assert isinstance(grid, CenteredGrid)
    kernel = math.zeros((3, 3, 1, 2))
    kernel[1, :, 0, 0] = [0, 1, -1]  # y-component: - dz/dx
    kernel[:, 1, 0, 1] = [0, -1, 1]  # x-component: dz/dy
    scalar_potential = grid.padded([pad_width, pad_width]).values
    vector_field = math.conv(scalar_potential, kernel, padding='valid')
    return StaggeredGrid(vector_field, bounds=grid.box)


def where(mask: Field or Geometry, field_true: Field, field_false: Field):
    if isinstance(mask, Geometry):
        mask = HardGeometryMask(mask)
    elif isinstance(mask, SampledField):
        field_true = field_true.at(mask)
        field_false = field_false.at(mask)
    elif isinstance(field_true, SampledField):
        mask = mask.at(field_true)
        field_false = field_false.at(field_true)
    elif isinstance(field_false, SampledField):
        mask = mask.at(field_true)
        field_true = field_true.at(mask)
    else:
        raise NotImplementedError('At least one argument must be a SampledField')
    values = mask.values * field_true.values + (1 - mask.values) * field_false.values
    # values = math.where(mask.values, field_true.values, field_false.values)
    return field_true._with(values)


def l2_loss(field: SampledField, batch_norm=True):
    return math.l2_loss(field.values, batch_norm=batch_norm)
