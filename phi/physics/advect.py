"""
Container for different advection schemes for grids and particles.

Examples:

* semi_lagrangian (grid)
* mac_cormack (grid)
* runge_kutta_4 (particle)
"""
import warnings

from phi import math
from phi.field import SampledField, ConstantField, Field, PointCloud, extrapolate_valid, Grid, sample, reduce_sample
from phi.field._field_math import GridType
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


def advect(field: SampledField,
           velocity: Field,
           dt: float or math.Tensor,
           integrator=euler) -> Field:
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

    Returns:
        Advected field of same type as `field`
    """
    if isinstance(field, PointCloud):
        return points(field, velocity, dt=dt, integrator=integrator)
    elif isinstance(field, Grid):
        return semi_lagrangian(field, velocity, dt=dt, integrator=integrator)
    elif isinstance(field, ConstantField):
        return field
    raise NotImplementedError(field)


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


def runge_kutta_4(cloud: SampledField, velocity: Field, dt: float, accessible: Field = None, occupied: Field = None):
    """ 
    Lagrangian advection of particles using a fourth-order runge-kutta scheme. If `accessible` and `occupied` are specified,
    the advection uses velocity-dependent extrapolation of `velocity`.
    
    Args:
        cloud: PointCloud holding the particle positions as elements
        velocity: velocity Grid which should get used for advection
        dt: Time step for runge-kutta
        accessible: Boundary conditions for restricting extrapolation to accessible positions
        occupied: Binary Grid indicating particle positions on the grid for extrapolation

    Returns:
        PointCloud with advected particle positions and their corresponding values.
    """
    warnings.warn("runge_kutta_4 is deprecated. Use points() with integrator=rk4 instead.", DeprecationWarning)
    assert isinstance(velocity, Grid), 'runge_kutta advection with extrapolation works for Grids only.'

    def extrapolation_helper(elements, t_shift, v_field, mask):
        shift = math.ceil(math.max(math.abs(elements.center - points.center))) - t_shift
        t_shift += shift
        v_field, mask = extrapolate_valid(v_field, mask, int(shift))
        v_field *= accessible
        return v_field, mask, t_shift

    points = cloud.elements
    total_shift = 0
    extrapolate = accessible is not None and occupied is not None

    # --- Sample velocity at intermediate points and adjust velocity-dependent
    # extrapolation to maximum shift of corresponding component ---
    if extrapolate:
        assert isinstance(occupied, type(velocity)), 'occupation mask must have same type as velocity.'
        velocity, occupied = extrapolate_valid(velocity, occupied, 2)
        velocity *= accessible
    vel_k1 = sample(velocity, points)

    shifted_points = points.shifted(0.5 * dt * vel_k1)
    if extrapolate:
        velocity, occupied, total_shift = extrapolation_helper(shifted_points, total_shift, velocity, occupied)
    vel_k2 = sample(velocity, shifted_points)

    shifted_points = points.shifted(0.5 * dt * vel_k2)
    if extrapolate:
        velocity, occupied, total_shift = extrapolation_helper(shifted_points, total_shift, velocity, occupied)
    vel_k3 = sample(velocity, shifted_points)

    shifted_points = points.shifted(dt * vel_k3)
    if extrapolate:
        velocity, _, _ = extrapolation_helper(shifted_points, total_shift, velocity, occupied)
    vel_k4 = sample(velocity, shifted_points)

    # --- Combine points with RK4 scheme ---
    vel = (1/6.) * (vel_k1 + 2 * (vel_k2 + vel_k3) + vel_k4)
    new_points = points.shifted(dt * vel)
    return cloud.with_elements(new_points)
