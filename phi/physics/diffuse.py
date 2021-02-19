"""
Functions to simulate diffusion processes on `phi.field.Field` objects.
"""
import warnings

from phi import math
from phi.field import ConstantField, Grid, Field, laplace, solve
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
        Diffused field of same type as `field`.
    """
    amount = diffusivity * dt
    if isinstance(amount, Field):
        amount = amount.at(field)
    for i in range(substeps):
        field += amount / substeps * laplace(field)
    return field


def implicit(field: FieldType,
             diffusivity: float or math.Tensor or Field,
             dt: float or math.Tensor,
             order: int = 1,
             solve_params: math.Solve = math.LinearSolve(bake='sparse')) -> FieldType:
    """
    Diffusion by solving a linear system of equations.

    Args:
        order: Order of method, 1=first order. This translates to `substeps` for the explicit sharpening.
        field:
        diffusivity: Diffusion per time. `diffusion_amount = diffusivity * dt`
        dt: Time interval. `diffusion_amount = diffusivity * dt`
        solve_params:

    Returns:
        Diffused field of same type as `field`.
    """
    def sharpen(x):
        return explicit(x, diffusivity, -dt, substeps=order)

    converged, diffused, iterations = solve(sharpen, field, field, solve_params=solve_params)
    if math.all_available(converged):
        assert converged, f"Implicit diffusion solve did not converge after {iterations} iterations. Last estimate: {diffused.values}"
    return diffused


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
        Diffused field of same type as `field`.
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
    result_k = math.fft(field.values) * diffuse_kernel
    result_values = math.real(math.ifft(result_k))
    return field.with_(values=result_values)
