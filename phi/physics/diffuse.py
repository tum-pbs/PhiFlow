"""
Functions to simulate diffusion processes on `phi.field.Field` objects.
"""
from typing import Union

from phi import math
from phi.field import Grid, Field, laplace, solve_linear, jit_compile_linear
from phi.field._field import FieldType
from phi.field._grid import GridType
from phiml.math import copy_with, shape, Solve


def explicit(field: FieldType,
             diffusivity: Union[float, math.Tensor, Field],
             dt: Union[float, math.Tensor],
             substeps: int = 1) -> FieldType:
    """
    Simulate a finite-time diffusion process of the form dF/dt = α · ΔF on a given `Field` FieldType with diffusion coefficient α.

    Args:
        field: CenteredGrid, StaggeredGrid or ConstantField
        diffusivity: Diffusion per time. `diffusion_amount = diffusivity * dt`
            Can be a number, `phiml.math.Tensor` or `phi.field.Field`.
            If a channel dimension is present, it will be interpreted as non-isotropic diffusion.
        dt: Time interval. `diffusion_amount = diffusivity * dt`
        substeps: number of iterations to use (Default value = 1)

    Returns:
        Diffused field of same type as `field`.
    """
    amount = diffusivity * dt / substeps
    if isinstance(amount, Field):
        amount = amount.at(field)
    ext = field.extrapolation
    for i in range(substeps):
        delta = laplace(field, weights=amount) if 'vector' in shape(amount) else amount * laplace(field)
        field = (field + delta.with_extrapolation(ext)).with_extrapolation(ext)
    return field


def implicit(field: FieldType,
             diffusivity: Union[float, math.Tensor, Field],
             dt: Union[float, math.Tensor],
             order: int = 1,
             solve=Solve('CG')) -> FieldType:
    """
    Diffusion by solving a linear system of equations.

    Args:
        order: Order of method, 1=first order. This translates to `substeps` for the explicit sharpening.
        field: `phi.field.Field` to diffuse.
        diffusivity: Diffusion per time. `diffusion_amount = diffusivity * dt`
        dt: Time interval. `diffusion_amount = diffusivity * dt`
        solve:

    Returns:
        Diffused field of same type as `field`.
    """
    @jit_compile_linear
    def sharpen(x):
        return explicit(x, diffusivity, -dt, substeps=order)

    if not solve.x0:
        solve = copy_with(solve, x0=field)
    return solve_linear(sharpen, y=field, solve=solve)


def finite_difference(grid: Grid,
                      diffusivity: Union[float, math.Tensor, Field],
                      order: int,
                      implicit: math.Solve) -> FieldType:

    """
    Diffusion by using a finite difference scheme.
    In contrast to `explicit` and `implicit` accuracy can be increased by using stencils of higher-order rather than calculating substeps.
    This is controlled by the `scheme` passed.

    Args:
        grid: CenteredGrid or StaggeredGrid
        diffusivity: Diffusion per time. `diffusion_amount = diffusivity * dt`
        order: Spatial order of accuracy.
            Higher orders entail larger stencils and more computation time but result in more accurate results assuming a large enough resolution.
            Supported: 2 explicit, 4 explicit, 6 implicit (inherited from `phi.field.laplace()`).
        implicit: When a `Solve` object is passed, performs an implicit operation with the specified solver and tolerances.
            Otherwise, an explicit stencil is used.

    Returns:
        Diffused grid of same type as `grid`.
    """
    diffusivity = diffusivity.at(grid) if isinstance(diffusivity, Field) else diffusivity
    return diffusivity * laplace(grid, order=order, implicit=implicit).with_extrapolation(grid.extrapolation)


def fourier(field: GridType,
            diffusivity: Union[float, math.Tensor],
            dt: Union[float, math.Tensor]) -> FieldType:
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
    assert isinstance(field, Grid), "Cannot diffuse field of type '%s'" % type(field)
    assert field.extrapolation == math.extrapolation.PERIODIC, "Fourier diffusion can only be applied to periodic fields."
    amount = diffusivity * dt
    k = math.fftfreq(field.resolution)
    k2 = math.vec_squared(k)
    fft_laplace = -(2 * math.PI) ** 2 * k2
    diffuse_kernel = math.exp(fft_laplace * amount)
    result_k = math.fft(field.values) * diffuse_kernel
    result_values = math.real(math.ifft(result_k))
    return field.with_values(result_values)
