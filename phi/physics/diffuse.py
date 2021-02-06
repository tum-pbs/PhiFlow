"""
Functions to simulate diffusion processes on `phi.field.Field` objects.
"""
from phi import math
from phi.field import ConstantField, Grid, Field, laplace
from phi.field._field_math import FieldType, GridType


def explicit(field: FieldType,
             diffusivity: float or math.Tensor or Field,
             dt: float or math.Tensor,
             substeps: int = 1) -> FieldType:
    """
    Simulate a finite-time diffusion process of the form dF/dt = α · ΔF on a given `Field` FieldType with diffusion coefficient α.

    If `field` is periodic (set via `extrapolation='periodic'`), diffusion may be simulated in Fourier space.
    Otherwise, finite differencing is used to approximate the

    Args:
      field: CenteredGrid, StaggeredGrid or ConstantField
      diffusivity: Diffusion per time. `diffusion_amount = diffusivity * dt`
      dt: Time interval. `diffusion_amount = diffusivity * dt`
      substeps: number of iterations to use (Default value = 1)
      field: FieldType:

    Returns:
      Field of same type as `field`
    """
    amount = diffusivity * dt
    if isinstance(amount, Field):
        amount = amount.at(field)
    for i in range(substeps):
        field += amount / substeps * laplace(field)
    return field


# def implicit(field: FieldType,
#              diffusivity: float or math.Tensor or Field,
#              dt: float or math.Tensor,
#              solve_params: math.LinearSolve) -> FieldType:
#     """ Not Implemented. """
#     raise NotImplementedError()


def fourier(field: GridType,
            diffusivity: float or math.Tensor,
            dt: float or math.Tensor) -> FieldType:
    """
    Exact diffusion of a periodic field in frequency space.

    For non-periodic fields or non-constant diffusivity, use another diffusion function such as `explicit()`.

    Args:
        field:
        diffusivity: Diffusion per time. `diffusion_amount = diffusivity * dt`
        dt: Time interval. `diffusion_amount = diffusivity * dt`

    Returns:
      Field of same type as `field`
    """
    if isinstance(field, ConstantField):
        return field
    assert isinstance(field, Grid), "Cannot diffuse field of type '%s'" % type(field)
    assert field.extrapolation == math.extrapolation.PERIODIC, "Fourier diffusion can only be applied to periodic fields."
    amount = diffusivity * dt
    k = math.fftfreq(field.resolution)
    k2 = math.vec_squared(k)
    fft_laplace = -(2 * math.PI) ** 2 * k2
    diffuse_kernel = math.exp(fft_laplace * amount)
    result_values = math.real(math.ifft(math.fft(field.values) * diffuse_kernel))
    return field.with_(values=result_values)
