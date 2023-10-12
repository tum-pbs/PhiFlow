"""
Container for different advection schemes for grids and particles.

Examples:

* semi_lagrangian (grid)
* mac_cormack (grid)
* runge_kutta_4 (particle)
"""
from phi import math
from phi.field import Field, PointCloud, Grid, spatial_gradient, unstack, stack, resample, reduce_sample, sample
from phi.math import Solve, channel
from phiml.math import Tensor


def euler(data: Field, velocity: Field, dt: float, v0: Tensor = None) -> Tensor:
    """ Euler integrator. """
    if v0 is None:
        v0 = sample(velocity, data.geometry, at=data.sampled_at, boundary=data.boundary)
    return data.points + v0 * dt


def rk4(data: Field, velocity: Field, dt: float, v0: Tensor = None) -> Tensor:
    """ Runge-Kutta-4 integrator. """
    if v0 is None:
        v0 = sample(velocity, data.geometry, at=data.sampled_at, boundary=data.boundary)
    v_half = sample(velocity, data.points + 0.5 * dt * v0, at=data.sampled_at, boundary=data.boundary)
    v_half2 = sample(velocity, data.points + 0.5 * dt * v_half, at=data.sampled_at, boundary=data.boundary)
    v_full = sample(velocity, data.points + dt * v_half2, at=data.sampled_at, boundary=data.boundary)
    v_rk4 = (1 / 6.) * (v0 + 2 * (v_half + v_half2) + v_full)
    return data.points + dt * v_rk4


def finite_rk4(data: Field, velocity: Grid, dt: float, v0: math.Tensor = None) -> Tensor:
    """ Runge-Kutta-4 integrator with Euler fallback where velocity values are NaN. """
    if v0 is None:
        v0 = sample(velocity, data.geometry, at=data.sampled_at, boundary=data.boundary)
    v_half = sample(velocity, data.points + 0.5 * dt * v0, at=data.sampled_at, boundary=data.boundary)
    v_half2 = sample(velocity, data.points + 0.5 * dt * v_half, at=data.sampled_at, boundary=data.boundary)
    v_full = sample(velocity, data.points + dt * v_half2, at=data.sampled_at, boundary=data.boundary)
    v_rk4 = (1 / 6.) * (v0 + 2 * (v_half + v_half2) + v_full)
    v_nan = math.where(math.is_finite(v_rk4), v_rk4, v0)
    return data.points + dt * v_nan


def advect(field: Field,
           velocity: Field,
           dt: float or math.Tensor,
           integrator=euler) -> Field:
    """
    Advect `field` along the `velocity` vectors using the specified integrator.

    The behavior depends on the type of `field`:

    * `phi.field.PointCloud`: Points are advected forward, see `points`.
    * `phi.field.Grid`: Sample points are traced backward, see `semi_lagrangian`.

    Args:
        field: Field to be advected as `phi.field.Field`.
        velocity: Any `phi.field.Field` that can be sampled in the elements of `field`.
        dt: Time increment
        integrator: ODE integrator for solving the movement.

    Returns:
        Advected field of same type as `field`
    """
    if field.is_point_cloud:
        return points(field, velocity, dt=dt, integrator=integrator)
    elif field.is_grid:
        return semi_lagrangian(field, velocity, dt=dt, integrator=integrator)
    raise NotImplementedError(field)


def finite_difference(grid: Grid,
                      velocity: Field,
                      order=2,
                      implicit: Solve = None) -> Field:

    """
    Finite difference advection using the differentiation Scheme indicated by `scheme` and a simple Euler step

    Args:
        grid: Grid to be advected
        velocity: `Grid` that can be sampled in the elements of `grid`.
        order: Spatial order of accuracy.
            Higher orders entail larger stencils and more computation time but result in more accurate results assuming a large enough resolution.
            Supported: 2 explicit, 4 explicit, 6 implicit (inherited from `phi.field.spatial_gradient()` and resampling).
            Passing order=4 currently uses 2nd-order resampling. This is work-in-progress.
        implicit: When a `Solve` object is passed, performs an implicit operation with the specified solver and tolerances.
            Otherwise, an explicit stencil is used.

    Returns:
        Advected grid of same type as `grid`
    """
    if grid.is_grid and grid.is_staggered:
        grad_list = [spatial_gradient(field_component, stack_dim=channel('gradient'), order=order, implicit=implicit) for field_component in grid.vector]
        grad_grid = grid.with_values(math.stack([component.values for component in grad_list], channel(velocity)))
        if order == 4:
            amounts = [grad * vel.at(grad, order=2) for grad, vel in zip(grad_grid.gradient, velocity.vector)]  # ToDo resampling does not yet support order=4
        else:
            velocity.vector[0].at(grad_grid.gradient[0], order=order, implicit=implicit)
            amounts = [grad * vel.at(grad, order=order, implicit=implicit) for grad, vel in zip(grad_grid.gradient, velocity.vector)]
        amount = sum(amounts)
    else:
        assert grid.is_grid and grid.is_centered, f"grid must be CenteredGrid or StaggeredGrid but got {type(grid)}"
        grad = spatial_gradient(grid, stack_dim=channel('gradient'), order=order, implicit=implicit)
        velocity = stack(unstack(velocity, dim='vector'), dim=channel('gradient'))
        amounts = velocity * grad
        amount = sum(amounts.gradient)
    return - amount


def points(field: PointCloud, velocity: Field, dt: float, integrator=euler):
    """
    Advects the sample points of a point cloud using a simple Euler step.
    Each point moves by an amount equal to the local velocity times `dt`.

    Args:
        field: point cloud to be advected
        velocity: velocity sampled at the same points as the point cloud
        dt: Euler step time increment
        integrator: ODE integrator for solving the movement.

    Returns:
        Advected point cloud
    """
    new_elements = field.geometry.at(integrator(field, velocity, dt))
    return field.with_elements(new_elements)


def semi_lagrangian(field: Field,
                    velocity: Field,
                    dt: float,
                    integrator=euler) -> Field:
    """
    Semi-Lagrangian advection with simple backward lookup.
    
    This method samples the `velocity` at the grid points of `field`
    to determine the lookup location for each grid point by walking backwards along the velocity vectors.
    The new values are then determined by sampling `field` at these lookup locations.

    Args:
        field: quantity to be advected, stored on a grid (CenteredGrid or StaggeredGrid)
        velocity: vector field, need not be compatible with with `field`.
        dt: time increment
        integrator: ODE integrator for solving the movement.

    Returns:
        Field with same sample points as `field`

    """
    lookup = integrator(field, velocity, -dt)
    interpolated = reduce_sample(field, lookup)
    return field.with_values(interpolated)


def mac_cormack(field: Field,
                velocity: Field,
                dt: float,
                correction_strength=1.0,
                integrator=euler) -> Field:
    """
    MacCormack advection uses a forward and backward lookup to determine the first-order error of semi-Lagrangian advection.
    It then uses that error estimate to correct the field values.
    To avoid overshoots, the resulting value is bounded by the neighbouring grid cells of the backward lookup.

    Args:
        field: Field to be advected, one of `(CenteredGrid, StaggeredGrid)`
        velocity: Vector field, need not be sampled at same locations as `field`.
        dt: Time increment
        correction_strength: The estimated error is multiplied by this factor before being applied.
            The case correction_strength=0 equals semi-lagrangian advection. Set lower than 1.0 to avoid oscillations.
        integrator: ODE integrator for solving the movement.

    Returns:
        Advected field of type `type(field)`
    """
    v0 = resample(velocity, to=field).values
    points_bwd = integrator(field, velocity, -dt, v0=v0)
    points_fwd = integrator(field, velocity, dt, v0=v0)
    # --- forward+backward semi-Lagrangian advection ---
    fwd_adv = field.with_values(reduce_sample(field, points_bwd))
    bwd_adv = field.with_values(reduce_sample(fwd_adv, points_fwd))
    new_field = fwd_adv + correction_strength * 0.5 * (field - bwd_adv)
    # --- Clamp overshoots ---
    limits = field.closest_values(points_bwd)
    lower_limit = math.min(limits, [f'closest_{dim}' for dim in field.shape.spatial.names])
    upper_limit = math.max(limits, [f'closest_{dim}' for dim in field.shape.spatial.names])
    values_clamped = math.clip(new_field.values, lower_limit, upper_limit)
    return new_field.with_values(values_clamped)
