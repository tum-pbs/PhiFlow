import warnings
from functools import wraps

import numpy as np
from phi import math
from phi.geom import Box
from . import StaggeredGrid, ConstantField
from ._field import Field, SampledField
from ._grid import CenteredGrid, Grid
from ..math import tensor


def laplace(field: Grid, axes=None):
    result = field._op1(lambda tensor: math.laplace(tensor, dx=field.dx, padding=field.extrapolation, axes=axes))
    return result


def gradient(field: Grid, axes=None, difference='central'):
    if not physical_units or self.has_cubic_cells:
        data = math.gradient(self.values, dx=np.mean(self.dx), difference=difference, padding=_pad_mode(self.extrapolation))
        return self.copied_with(data=data, extrapolation=self.extrapolation.gradient(), flags=())
    else:
        raise NotImplementedError('Only cubic cells supported.')


def shift(grid: CenteredGrid, offsets: tuple, stack_dim='shift'):
    """ Wraps :func:`math.shift` for CenteredGrid. """
    data = math.shift(grid.values, offsets, padding=grid.extrapolation, stack_dim=stack_dim)
    return [CenteredGrid(data[i], grid.box, grid.extrapolation) for i in range(len(offsets))]


def staggered_gradient(field: CenteredGrid):
    return stagger(field, lambda lower, upper: (upper - lower) / field.dx)


def stagger(field: CenteredGrid, face_function=math.minimum):
    all_lower = []
    all_upper = []
    for dim in field.shape.spatial.names:
        all_upper.append(math.pad(field.values, {dim: (0, 1)}, field.extrapolation))
        all_lower.append(math.pad(field.values, {dim: (1, 0)}, field.extrapolation))
    all_upper = math.channel_stack(all_upper, 'vector')
    all_lower = math.channel_stack(all_lower, 'vector')
    result = face_function(all_lower, all_upper)
    return result


def divergence(field: Grid):
    if isinstance(field, StaggeredGrid):
        components = []
        for i, dim in enumerate(field.shape.spatial.names):
            div_dim = math.gradient(field.values.vector[i], dx=field.dx[i], difference='forward', padding=None, axes=[dim]).gradient[0]
            components.append(div_dim)
        data = math.sum(components, 0)
        return CenteredGrid(data, field.box, field.extrapolation.gradient())
    else:
        raise NotImplementedError(field)


def diffuse(field: Field, diffusivity, dt, substeps=1):
    """
    Simulate a finite-time diffusion process of the form dF/dt = α · ΔF on a given `Field` F with diffusion coefficient α.

    If `field` is periodic (set via `extrapolation='periodic'`), diffusion may be simulated in Fourier space.
    Otherwise, finite differencing is used to approximate the

    :param field: CenteredGrid, StaggeredGrid or ConstantField
    :param diffusivity: diffusion amount = diffusivity * dt
    :param dt: diffusion amount = diffusivity * dt
    :param substeps: number of iterations to use
    :return: Field of same type as `field`
    :rtype: Field
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


def conjugate_gradient(function, y: Grid, x0: Grid, relative_tolerance: float = 1e-5, absolute_tolerance: float = 0.0, max_iterations: int = 1000, gradient: str = 'implicit', callback=None, bake='sparse'):
    if callback is not None:
        def field_callback(x):
            x = x0._with(x)
            callback(x)
    else:
        field_callback = None

    data_function = expose_tensors(function, x0)
    converged, x, iterations = math.conjugate_gradient(data_function, y.values, x0.values, relative_tolerance, absolute_tolerance, max_iterations, gradient, field_callback, bake)
    return converged, x0._with(x), iterations


def expose_tensors(field_function, *proto_fields):
    @wraps(field_function)
    def wrapper(*field_data):
        fields = [proto._with(data) for data, proto in zip(field_data, proto_fields)]
        return field_function(*fields).values
    return wrapper


def data_bounds(field: SampledField):
    data = field.points
    min_vec = math.min(data, axis=data.shape.spatial.names)
    max_vec = math.max(data, axis=data.shape.spatial.names)
    return Box(min_vec, max_vec)


def mean(field: Grid):
    return math.mean(field.values, field.shape.spatial)


def normalize(field: SampledField, norm: SampledField, epsilon=1e-5):
    data = math.normalize_to(field.values, norm.values, epsilon)
    return field._with(data)


def pad(grid: Grid, widths):
    if isinstance(widths, int):
        widths = {axis: (widths, widths) for axis in grid.shape.spatial.names}
    elif isinstance(widths, (tuple, list)):
        widths = {axis: (width if isinstance(width, (tuple, list)) else (width, width)) for axis, width in zip(grid.shape.spatial.names, widths)}
    else:
        assert isinstance(widths, dict)
    widths_list = [widths[axis] for axis in grid.shape.spatial.names]
    if isinstance(grid, CenteredGrid):
        data = math.pad(grid.values, widths, grid.extrapolation)
        w_lower = tensor([w[0] for w in widths_list])
        w_upper = tensor([w[1] for w in widths_list])
        box = Box(grid.box.lower - w_lower * grid.dx, grid.box.upper + w_upper * grid.dx)
        return CenteredGrid(data, box, grid.extrapolation)
    raise NotImplementedError()


def divergence_free(vector_field: Grid, relative_tolerance: float = 1e-3, absolute_tolerance: float = 0.0, max_iterations: int = 1000, bake='sparse'):
    """
    Returns the divergence-free part of the given vector field.
    The boundary conditions are taken from `vector_field`.

    This function solves for a scalar potential with an interative solver.

    :param vector_field: vector grid
    :param relative_tolerance: for the potential solver
    :param absolute_tolerance: for the potential solver
    :param max_iterations: for the potential solver
    :param bake: for the potential solver
    :return: divergence-free vector field, scalar potential, number of iterations performed, divergence
    """
    div = divergence(vector_field)
    div -= mean(div)
    pressure_extrapolation = vector_field.extrapolation  # periodic -> periodic, closed -> boundary, open -> zero
    pressure_guess = CenteredGrid.sample(0, vector_field.resolution, vector_field.box, extrapolation=pressure_extrapolation)
    converged, potential, iterations = conjugate_gradient(laplace, div, pressure_guess, relative_tolerance, absolute_tolerance, max_iterations, bake=bake)
    gradp = staggered_gradient(potential)
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
    return StaggeredGrid(vector_field, box=grid.box)
