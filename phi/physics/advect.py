"""
Container for different advection schemes for grids and particles.

Esamples:

* semi_lagrangian (grid)
* mac_cormack (grid)
* runge_kutta_4 (particle)
"""

from phi import math
from phi.field import SampledField, ConstantField, StaggeredGrid, CenteredGrid, Field, PointCloud, extrapolate_valid, Grid
from phi.field._field_math import GridType


def advect(field: Field, velocity: Field, dt, mode: str = 'euler', valid: Field = None, occupied: Field = None):
    """
    Advect `field` along the `velocity` vectors using the default advection method.

    Args:
      field: any built-in Field
      velocity: any Field (must be PointCloud with the same elements as `field` in euler mode or
        a StaggeredGrid in rk4_extp mode)
      dt: time increment
      mode: type of advection scheme: 'euler', 'rk4'
      valid: boundary conditions used only in 'rk4' mode
      occupied: binary field of the same type as velocity indicating particle positions (only used in 'rk4' modes)

    Returns:
      Advected field of same type as `field`
    """
    if isinstance(field, PointCloud):
        if mode == 'euler':
            assert isinstance(velocity, PointCloud) and velocity.elements == field.elements, 'Velocity is not valid for euler mode advection.'
            return points(field, velocity, dt)
        elif mode == 'rk4':
            return runge_kutta_4(field, velocity, dt=dt, accessible=valid, occupied=occupied)
        else:
            raise NotImplementedError(f"Advection mode {mode} is not known.")
    if isinstance(field, ConstantField):
        return field
    if isinstance(field, (CenteredGrid, StaggeredGrid)):
        return semi_lagrangian(field, velocity, dt=dt)
    raise NotImplementedError(field)


def semi_lagrangian(field: GridType, velocity: Field, dt) -> GridType:
    """
    Semi-Lagrangian advection with simple backward lookup.
    
    This method samples the `velocity` at the grid points of `field`
    to determine the lookup location for each grid point by walking backwards along the velocity vectors.
    The new values are then determined by sampling `field` at these lookup locations.

    Args:
      field: quantity to be advected, stored on a grid (CenteredGrid or StaggeredGrid)
      velocity: vector field, need not be compatible with with `field`.
      dt: time increment
      field: GridType: 
      velocity: Field: 

    Returns:
      Field with same sample points as `field`

    """
    v = velocity.sample_in(field.elements)
    x = field.points - v * dt
    interpolated = field.sample_at(x, reduce_channels=x.shape.non_channel.without(field.shape).names)
    return field.with_(values=interpolated)


def mac_cormack(field: GridType, velocity: Field, dt: float, correction_strength=1.0) -> GridType:
    """
    MacCormack advection uses a forward and backward lookup to determine the first-order error of semi-Lagrangian advection.
    It then uses that error estimate to correct the field values.
    To avoid overshoots, the resulting value is bounded by the neighbouring grid cells of the backward lookup.

    Args:
      field: Field to be advected, one of `(CenteredGrid, StaggeredGrid)`
      velocity: Vector field, need not be sampled at same locations as `field`.
      dt: Time increment
      correction_strength: The estimated error is multiplied by this factor before being applied. The case correction_strength=0 equals semi-lagrangian advection. Set lower than 1.0 to avoid oscillations. (Default value = 1.0)

    Returns:
      Advected field of type `type(field)`

    """
    v = velocity.sample_in(field.elements)
    x0 = field.points
    x_bwd = x0 - v * dt
    x_fwd = x0 + v * dt
    reduce = x0.shape.non_channel.without(field.shape).names
    # Semi-Lagrangian advection
    field_semi_la = field.with_(values=field.sample_at(x_bwd, reduce_channels=reduce))
    # Inverse semi-Lagrangian advection
    field_inv_semi_la = field.with_(values=field_semi_la.sample_at(x_fwd, reduce_channels=reduce))
    # correction
    new_field = field_semi_la + correction_strength * 0.5 * (field - field_inv_semi_la)
    # Address overshoots
    limits = field.closest_values(x_bwd, reduce_channels=reduce)
    lower_limit = math.min(limits, [f'closest_{dim}' for dim in field.shape.spatial.names])
    upper_limit = math.max(limits, [f'closest_{dim}' for dim in field.shape.spatial.names])
    values_clamped = math.clip(new_field.values, lower_limit, upper_limit)
    return new_field.with_(values=values_clamped)


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
    assert isinstance(velocity, Grid), 'runge_kutta advection with extrapolation works for Grids only.'
    assert isinstance(occupied, type(velocity)), 'occupation mask must have same type as velocity.'

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
        velocity, occupied = extrapolate_valid(velocity, occupied, 2)
        velocity *= accessible
    vel_k1 = velocity.sample_in(points)

    shifted_points = points.shifted(0.5 * dt * vel_k1)
    if extrapolate:
        velocity, occupied, total_shift = extrapolation_helper(shifted_points, total_shift, velocity, occupied)
    vel_k2 = velocity.sample_in(shifted_points)

    shifted_points = points.shifted(0.5 * dt * vel_k2)
    if extrapolate:
        velocity, occupied, total_shift = extrapolation_helper(shifted_points, total_shift, velocity, occupied)
    vel_k3 = velocity.sample_in(shifted_points)

    shifted_points = points.shifted(dt * vel_k3)
    if extrapolate:
        velocity, _, _ = extrapolation_helper(shifted_points, total_shift, velocity, occupied)
    vel_k4 = velocity.sample_in(shifted_points)

    # --- Combine points with RK4 scheme ---
    vel = (1/6.) * (vel_k1 + 2 * (vel_k2 + vel_k3) + vel_k4)
    new_points = points.shifted(dt * vel)
    return cloud.with_(elements=new_points)


def points(field: PointCloud, velocity: PointCloud, dt):
    """
    Advects the sample points of a point cloud using a simple Euler step.
    Each point moves by an amount equal to the local velocity times `dt`.

    Args:
      field: point cloud to be advected
      velocity: velocity sampled at the same points as the point cloud
      dt: Euler step time increment
      field: PointCloud: 
      velocity: PointCloud: 

    Returns:
      advected point cloud

    """
    assert field.elements == velocity.elements
    new_points = field.elements.shifted(dt * velocity.values)
    return field.with_(elements=new_points)
