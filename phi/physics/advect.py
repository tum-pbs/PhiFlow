"""
Container for different advection schemes for grids and particles.

Examples:

* semi_lagrangian (grid)
* mac_cormack (grid)
* runge_kutta_4 (particle)
"""
from phi import math
from phi.field import SampledField, Field, PointCloud, Grid, sample, reduce_sample, spatial_gradient, unstack, stack, CenteredGrid, StaggeredGrid
from phi.field._field import FieldType
from phi.field._field_math import GridType
from phi.field.numerical import Scheme
from phi.geom import Geometry


def euler(elements: Geometry, velocity: Field, dt: float, v0: math.Tensor = None) -> Geometry:
    """ Euler integrator. """
    if v0 is None:
        v0 = sample(velocity, elements)
    return elements.shifted(v0 * dt)


def rk4(elements: Geometry, velocity: Field, dt: float, v0: math.Tensor = None) -> Geometry:
    """ Runge-Kutta-4 integrator. """
    if v0 is None:
        v0 = sample(velocity, elements)
    vel_half = sample(velocity, elements.shifted(0.5 * dt * v0))
    vel_half2 = sample(velocity, elements.shifted(0.5 * dt * vel_half))
    vel_full = sample(velocity, elements.shifted(dt * vel_half2))
    vel_rk4 = (1 / 6.) * (v0 + 2 * (vel_half + vel_half2) + vel_full)
    return elements.shifted(dt * vel_rk4)


def finite_rk4(elements: Geometry, velocity: Grid, dt: float, v0: math.Tensor = None) -> Geometry:
    """ Runge-Kutta-4 integrator with Euler fallback where velocity values are NaN. """
    v0 = sample(velocity, elements)
    vel_half = sample(velocity, elements.shifted(0.5 * dt * v0))
    vel_half2 = sample(velocity, elements.shifted(0.5 * dt * vel_half))
    vel_full = sample(velocity, elements.shifted(dt * vel_half2))
    vel_rk4 = (1 / 6.) * (v0 + 2 * (vel_half + vel_half2) + vel_full)
    vel_nan = math.where(math.is_finite(vel_rk4), vel_rk4, v0)
    return elements.shifted(dt * vel_nan)



def advect(field: SampledField,
           velocity: Field,
           dt: float or math.Tensor,
           integrator=euler,
           scheme: Scheme = None) -> FieldType:
    """
    Advect `field` along the `velocity` vectors using the specified integrator.

    The behavior depends on the type of `field`:

    * `phi.field.PointCloud`: Points are advected forward, see `points`.
    * `phi.field.Grid`: Sample points are traced backward, see `semi_lagrangian`.

    Args:
        field: Field to be advected as `phi.field.SampledField`.
        velocity: Any `phi.field.Field` that can be sampled in the elements of `field`.
        dt: Time increment
        integrator: ODE integrator for solving the movement.
        scheme: differentiation 'Scheme' if provided 'finite_difference' is used
            if 'None' is given other functions are used which is the case by default

    Returns:
        Advected field of same type as `field`
    """

    if scheme is not None and isinstance(field, Grid):
        return finite_difference(field, velocity, dt=dt, scheme=scheme)
    if isinstance(field, PointCloud):
        return points(field, velocity, dt=dt, integrator=integrator)
    elif isinstance(field, Grid):
        return semi_lagrangian(field, velocity, dt=dt, integrator=integrator)
    raise NotImplementedError(field)


def finite_difference(grid: Grid,
                      velocity: Field,
                      dt: float or math.Tensor,
                      scheme: Scheme = Scheme(2)) -> Field:

    """
    Finite difference advection using the differentiation Scheme indicated by `scheme` and a simple Euler step

    Args:
        grid: Grid to be advected
        velocity: `Grid` that can be sampled in the elements of `grid`.
        dt: Time increment
        scheme: finite difference `Scheme` used for differentiation
            supported: explicit 2/4th order - implicit 6th order

    Returns:
        Advected grid of same type as `grid`
    """

    if isinstance(grid, StaggeredGrid):
        field_components = unstack(grid, 'vector')
        grad_list = [spatial_gradient(field_component, stack_dim=math.channel('gradient'), scheme=scheme) for
                     field_component in field_components]
        grad_grid = grid.with_values(math.stack([component.values for component in grad_list], math.channel('vector')))
        velocity._scheme = True
        ammounts = [grad * vel.at(grad, scheme=scheme) for grad, vel in
                    zip(unstack(grad_grid, dim='gradient'), unstack(velocity, dim='vector'))]
        ammount = sum(ammounts)
    else:
        grad = spatial_gradient(grid, stack_dim=math.channel('gradient'), scheme=scheme)
        velocity = stack(unstack(velocity, dim='vector'), dim=math.channel('gradient'))
        ammounts = velocity * grad
        ammount = sum(unstack(ammounts, dim='gradient'))

    return grid - dt * ammount


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
    new_elements = integrator(field.elements, velocity, dt)
    return field.with_elements(new_elements)


def semi_lagrangian(field: GridType,
                    velocity: Field,
                    dt: float,
                    integrator=euler) -> GridType:
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
    lookup = integrator(field.elements, velocity, -dt)
    interpolated = reduce_sample(field, lookup)
    return field.with_values(interpolated)


def mac_cormack(field: GridType,
                velocity: Field,
                dt: float,
                correction_strength=1.0,
                integrator=euler) -> GridType:
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
    v0 = sample(velocity, field.elements)
    points_bwd = integrator(field.elements, velocity, -dt, v0=v0)
    points_fwd = integrator(field.elements, velocity, dt, v0=v0)
    # Semi-Lagrangian advection
    field_semi_la = field.with_values(reduce_sample(field, points_bwd))
    # Inverse semi-Lagrangian advection
    field_inv_semi_la = field.with_values(reduce_sample(field_semi_la, points_fwd))
    # correction
    new_field = field_semi_la + correction_strength * 0.5 * (field - field_inv_semi_la)
    # Address overshoots
    limits = field.closest_values(points_bwd)
    lower_limit = math.min(limits, [f'closest_{dim}' for dim in field.shape.spatial.names])
    upper_limit = math.max(limits, [f'closest_{dim}' for dim in field.shape.spatial.names])
    values_clamped = math.clip(new_field.values, lower_limit, upper_limit)
    return new_field.with_values(values_clamped)
