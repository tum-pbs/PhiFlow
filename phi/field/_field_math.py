from functools import wraps

import numpy as np
from phi import math
from phi.geom import Box
from . import StaggeredGrid
from ._field import Field, SampledField
from ._grid import CenteredGrid, Grid, _gradient_extrapolation
from ..math import tensor


def laplace(field: Grid, axes=None):
    result = field._op1(lambda tensor: math.laplace(tensor, dx=field.dx, padding=field.extrapolation, axes=axes))
    return result


def gradient(field: Grid, axes=None, difference='central'):
    if not physical_units or self.has_cubic_cells:
        data = math.gradient(self.data, dx=np.mean(self.dx), difference=difference, padding=_pad_mode(self.extrapolation))
        return self.copied_with(data=data, extrapolation=_gradient_extrapolation(self.extrapolation), flags=())
    else:
        raise NotImplementedError('Only cubic cells supported.')


def shift(grid: CenteredGrid, offsets: tuple, stack_dim='shift'):
    data = math.shift(grid.data, offsets, padding=grid.extrapolation, stack_dim=stack_dim)
    return [CenteredGrid(data[i], grid.box, grid.extrapolation) for i in range(len(offsets))]


def staggered_gradient(field: CenteredGrid):
    return stagger(field, lambda lower, upper: (upper - lower) / field.dx)


def stagger(field: CenteredGrid, face_function=math.minimum):
    all_lower = []
    all_upper = []
    for dim in field.shape.spatial.names:
        all_upper.append(math.pad(field.data, {dim: (0, 1)}, field.extrapolation))
        all_lower.append(math.pad(field.data, {dim: (1, 0)}, field.extrapolation))
    all_upper = math.channel_stack(all_upper, 'vector')
    all_lower = math.channel_stack(all_lower, 'vector')
    result = face_function(all_lower, all_upper)
    return result


def divergence(field: Grid):
    if isinstance(field, StaggeredGrid):
        components = []
        for i, dim in enumerate(field.shape.spatial.names):
            div_dim = math.gradient(field.data.vector[i], dx=field.dx[i], difference='forward', padding=None, axes=[dim]).gradient[0]
            components.append(div_dim)
        data = math.sum(components, 0)
        return CenteredGrid(data, field.box, field.extrapolation.gradient())
    else:
        raise NotImplementedError(field)


def conjugate_gradient(function, y: Grid, x0: Grid, relative_tolerance: float = 1e-5, absolute_tolerance: float = 0.0, max_iterations: int = 1000, gradient: str = 'implicit', callback=None, bake='sparse'):
    if callback is not None:
        def field_callback(x):
            x = x0.with_data(x)
            callback(x)
    else:
        field_callback = None

    data_function = expose_tensors(function, x0)
    converged, x, iterations = math.conjugate_gradient(data_function, y.data, x0.data, relative_tolerance, absolute_tolerance, max_iterations, gradient, field_callback, bake)
    return converged, x0.with_data(x), iterations


def expose_tensors(field_function, *proto_fields):
    @wraps(field_function)
    def wrapper(*field_data):
        fields = [proto.with_data(data) for data, proto in zip(field_data, proto_fields)]
        return field_function(*fields).data
    return wrapper


def data_bounds(field: SampledField):
    data = field.points
    min_vec = math.min(data, axis=data.shape.spatial.names)
    max_vec = math.max(data, axis=data.shape.spatial.names)
    return Box(min_vec, max_vec)



def mean(field: Grid):
    return math.mean(field.data, field.shape.spatial)


def normalize(field: SampledField, norm: SampledField, epsilon=1e-5):
    data = math.normalize_to(field.data, norm.data, epsilon)
    return field.with_data(data)


def pad(grid: Grid, widths):
    if isinstance(widths, int):
        widths = {axis: (widths, widths) for axis in grid.shape.spatial.names}
    elif isinstance(widths, (tuple, list)):
        widths = {axis: (width if isinstance(width, (tuple, list)) else (width, width)) for axis, width in zip(grid.shape.spatial.names, widths)}
    else:
        assert isinstance(widths, dict)
    widths_list = [widths[axis] for axis in grid.shape.spatial.names]
    if isinstance(grid, CenteredGrid):
        data = math.pad(grid.data, widths, grid.extrapolation)
        w_lower = tensor([w[0] for w in widths_list])
        w_upper = tensor([w[1] for w in widths_list])
        box = Box(grid.box.lower - w_lower * grid.dx, grid.box.upper + w_upper * grid.dx)
        return CenteredGrid(data, box, grid.extrapolation)
    raise NotImplementedError()


def divergence_free(velocity: Grid, relative_tolerance: float = 1e-3, absolute_tolerance: float = 0.0, max_iterations: int = 1000, bake: str = 'sparse'):
    """

    :param velocity:
    :param relative_tolerance:
    :param absolute_tolerance:
    :param max_iterations:
    :param bake:
    :return: divergence-free velocity, pressure, iterations, divergence
    """
    # TODO do we need the domain? closed -> boundary extrapolation
    div = divergence(velocity)
    div -= mean(div)
    pressure_extrapolation = velocity.extrapolation  # TODO periodic -> periodic, closed -> zero-grdient, open -> ?
    pressure_guess = CenteredGrid.sample(0, velocity.resolution, velocity.box, extrapolation=pressure_extrapolation)
    converged, pressure, iterations = conjugate_gradient(laplace, div, pressure_guess, relative_tolerance, absolute_tolerance, max_iterations, bake=bake)
    gradp = staggered_gradient(pressure)
    velocity -= gradp
    return velocity, pressure, iterations, div


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
    scalar_potential = grid.padded([pad_width, pad_width]).data
    vector_field = math.conv(scalar_potential, kernel, padding='valid')
    return StaggeredGrid(vector_field, box=grid.box)
