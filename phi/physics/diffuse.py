"""
Functions to simulate diffusion processes on `phi.field.Field` objects.
"""
import warnings
from typing import Union

from phi import math
from phi.field import Grid, Field, laplace, solve_linear, jit_compile_linear
from phiml.math import copy_with, shape, Solve, wrap, spatial, Tensor, dual, rename_dims
from phiml.math.extrapolation import NONE


def explicit(field: Field,
             diffusivity: Union[float, Tensor, Field],
             dt: Union[float, Tensor],
             substeps: int = 1) -> Field:
    """
    Simulate a finite-time diffusion process of the form dF/dt = α · ΔF on a given `Field` Field with diffusion coefficient α.

    Args:
        field: CenteredGrid, StaggeredGrid or ConstantField
        diffusivity: Diffusion per time. `diffusion_amount = diffusivity * dt`
            Can be a number, `phi.Tensor` or `phi.field.Field`.
            If a channel dimension is present, it will be interpreted as non-isotropic diffusion.
        dt: Time interval. `diffusion_amount = diffusivity * dt`
        substeps: number of iterations to use (Default value = 1)

    Returns:
        Diffused field of same type as `field`.
    """
    amount = diffusivity * (dt / substeps)
    # --- CFL check if possible ---
    amount_ = amount.values if isinstance(amount, Field) else wrap(amount)
    if amount_.available:
        cfl = math.max(amount_, spatial) / field.dx**2
        if (cfl > .5).any:
            warnings.warn(f"CFL condition violated (CFL = {float(cfl.max):.1f} > 0.5) in diffuse.explicit() with diffusivity={diffusivity}, dt={dt}, dx={field.dx}. Increase substeps or use diffuse.implicit() instead.", RuntimeWarning, stacklevel=2)
    # --- diffusion ---
    if isinstance(amount, Field):
        amount = amount.at(field)
    ext = field.extrapolation
    for i in range(substeps):
        delta = laplace(field, weights=amount) if 'vector' in shape(amount) else amount * laplace(field)
        field = (field + delta.with_extrapolation(ext)).with_extrapolation(ext)
    return field


def implicit(field: Field,
             diffusivity: Union[float, Tensor, Field],
             dt: Union[float, Tensor],
             order: int = 1,
             solve=Solve('CG')) -> Field:
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


def differential(u: Field,
                 diffusivity: Union[float, math.Tensor, Field],
                 gradient: Field = None,
                 order: int = 2,
                 implicit: math.Solve = None,
                 upwind: Field = None) -> Field:
    """
    Compute the differential diffusion term, d·∇²u.
    For grids, uses a finite difference scheme specified by `order` and `implicit`.
    For FVM, the scheme is specified via `order` and `upwind`.

    In contrast to `explicit` and `implicit`, accuracy can be increased by using stencils of higher-order rather than calculating sub-steps.

    Args:
        u: Scalar or vector-valued `Field` sampled on a `CenteredGrid`, `StaggeredGrid` or centered `UnstructuredMesh`.
        diffusivity: Dynamic viscosity, i.e. diffusion per time. Constant or varying by cell.
        gradient: Only used by FVM at the moment. Approximate gradient of `u`, e.g. ∇u of the previous time step.
            If `None`, approximates the gradient as `(u_neighbor - u_self) / distance`.
        order: Spatial order of accuracy.
            Higher orders entail larger stencils and more computation time but result in more accurate results assuming a large enough resolution.
            Supported: 2 explicit, 4 explicit, 6 implicit (inherited from `phi.field.laplace()`).
            For FVM, the order is used when interpolating `v` and `prev_v` to cell faces if needed.
        implicit: When a `Solve` object is passed, performs an implicit operation with the specified solver and tolerances.
            Otherwise, an explicit stencil is used.
        upwind: FVM only. Whether to use upwind interpolation.

    Returns:
        Differential diffusion as a `Field` on the same geometry.
    """
    if u.is_grid:
        diffusivity = diffusivity.at(u) if isinstance(diffusivity, Field) else diffusivity
        return diffusivity * laplace(u, order=order, implicit=implicit).with_extrapolation(u.boundary)
    elif u.is_mesh:
        neighbor_val = u.mesh.pad_boundary(u.values, mode=u.boundary)
        nb_distances = u.mesh.neighbor_distances
        connecting_grad = (u.mesh.connectivity * neighbor_val - u.values) / nb_distances  # (T_N - T_P) / d_PN
        if gradient is not None:  # skewness correction
            assert dual(gradient), f"prev_grad must contain a dual dimension listing the gradient components"
            gradient = rename_dims(gradient, dual, 'vector')
            gradient = gradient.at_faces(boundary=NONE, order=order, upwind=upwind).values
            nb_offsets = u.mesh.neighbor_offsets
            n1 = (u.face_normals.vector @ nb_offsets.vector) * nb_offsets / nb_distances ** 2  # (n·d_PN) d_PN / d_PN^2
            n2 = u.face_normals - n1
            ortho_correction = gradient @ n2
            grad = connecting_grad * math.vec_length(n1) + ortho_correction
        else:
            grad = connecting_grad
        return diffusivity * u.mesh.integrate_surface(grad) / u.mesh.volume  # 1/V ∑_f ∇T ν A
    raise NotImplementedError


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
