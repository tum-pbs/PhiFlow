"""
Functions to simulate diffusion processes on `phi.field.Field` objects.
"""
import warnings
from typing import Union

from phi import math
from phi.field import Grid, Field, laplace, solve_linear, jit_compile_linear, stagger
from phiml.math import copy_with, Solve, wrap, spatial, Tensor
from phiml.math.extrapolation import NONE


def explicit(u: Field,
             diffusivity: Union[float, Tensor, Field],
             dt: Union[float, Tensor],
             substeps: int = 1,
             order: int = 2,
             implicit: math.Solve = None,
             gradient: Field = None,
             upwind: Field = None,
             correct_skew=True) -> Field:
    """
    Explicit Euler diffusion with substeps.

    Simulate a finite-time diffusion process of the form dF/dt = α · ΔF on a given `Field` Field with diffusion coefficient α.

    Args:
        u: CenteredGrid, StaggeredGrid or ConstantField
        diffusivity: Diffusion per time. `diffusion_amount = diffusivity * dt`
            Can be a number, `phi.Tensor` or `phi.field.Field`.
            If a channel dimension is present, it will be interpreted as non-isotropic diffusion.
        dt: Time interval. `diffusion_amount = diffusivity * dt`
        substeps: number of iterations to use (Default value = 1)
        order: Spatial order of accuracy.
            Higher orders entail larger stencils and more computation time but result in more accurate results assuming a large enough resolution.
            Supported: 2 explicit, 4 explicit, 6 implicit (inherited from `phi.field.laplace()`).
            For FVM, the order is used when interpolating `v` and `prev_v` to cell faces if needed.
        implicit: When a `Solve` object is passed, performs a spatially implicit operation with the specified solver and tolerances.
            Otherwise, an explicit stencil is used.
        gradient: Only used by FVM at the moment. Approximate gradient of `u`, e.g. ∇u of the previous time step.
            If `None`, approximates the gradient as `(u_neighbor - u_self) / distance`.
        upwind: For unstructured meshes only. Whether to use upwind interpolation.
        correct_skew: If `True`, adds a correction term for cell skewness. This requires `gradient` to be passed.

    Returns:
        Diffused field of same type as `field`.
    """
    amount = diffusivity * (dt / substeps)
    # --- CFL check if possible ---
    amount_ = amount.values if isinstance(amount, Field) else wrap(amount)
    if amount_.available and u.is_grid:
        cfl = math.max(amount_, spatial) / u.dx ** 2
        if (cfl > .5).any:
            warnings.warn(f"CFL condition violated (CFL = {float(cfl.max):.1f} > 0.5) in diffuse.explicit() with diffusivity={diffusivity}, dt={dt}, dx={u.dx}. Increase substeps or use diffuse.implicit() instead.", RuntimeWarning, stacklevel=2)
    # --- diffusion ---
    if isinstance(amount, Field):
        amount = amount.at(u)
    for i in range(substeps):
        u += differential(u, amount, gradient=gradient, order=order, implicit=implicit, upwind=upwind, correct_skew=correct_skew)
    return u


def implicit(field: Field,
             diffusivity: Union[float, Tensor, Field],
             dt: Union[float, Tensor],
             solve=Solve('CG'),
             gradient: Field = None,
             upwind: Field = None,
             correct_skew=True,
             gradient_for_diffusivity=True) -> Field:
    """
    Implicit Euler diffusion.

    Diffusion by solving a linear system of equations.

    Args:
        field: `phi.field.Field` to diffuse.
        diffusivity: Diffusion per time. `diffusion_amount = diffusivity * dt`
        dt: Time interval. `diffusion_amount = diffusivity * dt`
        solve: Implicit solve parameters.
        gradient: Only used by FVM at the moment. Approximate gradient of `u`, e.g. ∇u of the previous time step.
            If `None`, approximates the gradient as `(u_neighbor - u_self) / distance`.
        upwind: For unstructured meshes only. Whether to use upwind interpolation.
        correct_skew: If `True`, adds a correction term for cell skewness. This requires `gradient` to be passed.
        gradient_for_diffusivity: Whether to compute the gradient w.r.t. the diffusivity parameters.

    Returns:
        Diffused field of same type as `field`.
    """
    @jit_compile_linear
    def sharpen(x):
        return explicit(x, diffusivity, -dt, gradient=gradient, upwind=upwind, correct_skew=correct_skew)
    if not solve.x0:
        solve = copy_with(solve, x0=field)
    return solve_linear(sharpen, y=field, solve=solve, grad_for_f=gradient_for_diffusivity)


def differential(u: Field,
                 diffusivity: Union[float, math.Tensor, Field],
                 gradient: Field = None,
                 order: int = 2,
                 implicit: math.Solve = None,
                 upwind: Field = None,
                 correct_skew=True) -> Field:
    """
    Compute the differential diffusion term, d·∇²u.
    For grids, uses a finite difference scheme specified by `order` and `implicit`.
    For FVM, the scheme is specified via `order` and `upwind`.

    In contrast to `explicit` and `implicit`, accuracy can be increased by using stencils of higher-order rather than calculating sub-steps.

    Args:
        u: Scalar or vector-valued `Field` sampled on a `CenteredGrid`, `StaggeredGrid` or centered `Mesh`.
        diffusivity: Dynamic viscosity, i.e. diffusion per time. Constant or varying by cell.
        gradient: Only used by FVM at the moment. Approximate gradient of `u`, e.g. ∇u of the previous time step.
            If `None`, approximates the gradient as `(u_neighbor - u_self) / distance`.
        order: Spatial order of accuracy.
            Higher orders entail larger stencils and more computation time but result in more accurate results assuming a large enough resolution.
            Supported: 2 explicit, 4 explicit, 6 implicit (inherited from `phi.field.laplace()`).
            For FVM, the order is used when interpolating `v` and `prev_v` to cell faces if needed.
        implicit: When a `Solve` object is passed, performs an implicit operation with the specified solver and tolerances.
            Otherwise, an explicit stencil is used.
        upwind: For unstructured meshes only. Whether to use upwind interpolation.
        correct_skew: If `True`, adds a correction term for cell skewness. This requires `gradient` to be passed.

    Returns:
        Differential diffusion as a `Field` on the same geometry.
    """
    if spatial(diffusivity):  # spatially-varying diffusivity
        assert order == 2, f"spatially-varying diffusivity only supported for second-order but got order={order}"
        # make sure outflow = neighbor inflow, i.e. make the matrix symmetric
        diffusivity: Field = diffusivity if isinstance(diffusivity, Field) else u.with_values(diffusivity)
        if u.is_grid and u.is_centered:
            face_diffusivity = stagger(diffusivity, math.minimum, NONE)
            du = u.gradient(boundary=NONE, at='face')
            lap = (face_diffusivity * du).divergence(order=2)
        else:
            raise NotImplementedError("spatially-varying diffusion currently only supported for centered grids")
    else:
        lap = laplace(u, weights=diffusivity, gradient=gradient, order=order, implicit=implicit, upwind=upwind, correct_skew=correct_skew)
    return lap.with_extrapolation(u.boundary - u.boundary)  # remove constants from extrapolation


finite_difference = differential


def fourier(field: Field,
            diffusivity: Union[float, Tensor],
            dt: Union[float, Tensor]) -> Field:
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
