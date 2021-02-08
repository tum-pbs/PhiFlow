"""
Container for different advection schemes for grids and particles.

Esamples:

* semi_lagrangian (grid)
* mac_cormack (grid)
* runge_kutta_4 (particle)
"""

from phi import math
from phi.field import SampledField, ConstantField, StaggeredGrid, CenteredGrid, Field, PointCloud, extp_sgrid
from phi.field._field_math import GridType


def advect(field: Field, velocity: Field, dt, mode: str = 'euler', bcs: Field = None, smask: Field = None):
    """
    Advect `field` along the `velocity` vectors using the default advection method.

    Args:
      field: any built-in Field
      velocity: any Field (must be PointCloud with the same elements as `field` in euler mode or
        a StaggeredGrid in rk4_extp mode)
      dt: time increment
      mode: type of advection scheme: 'euler', 'rk4' or 'rk4_extp'
      bcs: boundary conditions used only in 'rk4_extp' mode

    Returns:
      Advected field of same type as `field`

    """
    if isinstance(field, PointCloud):
        if mode == 'euler':
            assert isinstance(velocity, PointCloud) and velocity.elements == field.elements, 'Velocity is not valid for euler mode advection.'
            return points(field, velocity, dt)
        elif mode == 'rk4':
            return runge_kutta_4(field, velocity, dt=dt)
        elif mode == 'rk4_extp':
            return runge_kutta_4_extp(field, velocity, bcs=bcs, smask=smask, dt=dt)
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


def runge_kutta_4(field: PointCloud, velocity: Field, dt):
    """
    Lagrangian advection of particles.

    Args:
      field(SampledField): SampledField with any number of components
      velocity(Field): Vector field
      dt: time increment
      field: PointCloud: 
      velocity: Field: 

    Returns:
      SampledField with same data as `field` but advected points

    """
    assert isinstance(field, SampledField)
    assert isinstance(velocity, Field)
    points = field.elements
    # --- Sample velocity at intermediate points ---
    vel_k1 = velocity.sample_in(points)
    vel_k2 = velocity.sample_in(points.shifted(0.5 * dt * vel_k1))
    vel_k3 = velocity.sample_in(points.shifted(0.5 * dt * vel_k2))
    vel_k4 = velocity.sample_in(points.shifted(dt * vel_k3))
    # --- Combine points with RK4 scheme ---
    vel = (1/6.) * (vel_k1 + 2 * (vel_k2 + vel_k3) + vel_k4)
    new_points = points.shifted(dt * vel)
    return PointCloud(new_points, field.values, field.extrapolation, add_overlapping=field._add_overlapping,
                      bounds=field.bounds, color=field.color)


def runge_kutta_4_extp(field: PointCloud, velocity: Field, bcs: Field, smask: Field, dt: float):
    assert isinstance(field, SampledField)
    assert isinstance(velocity, StaggeredGrid), 'runge_kutta advection with extrapolation works for StaggeredGrids only.'
    assert isinstance(smask, StaggeredGrid), 'extrapolation mask must be of type StaggeredGrid'
    points = field.elements

    # --- Sample velocity at intermediate points ---
    # At first step all points are inside velocity region
    velocity, smask = extp_sgrid(velocity, smask, 2)
    velocity *= bcs
    vel_k1 = velocity.sample_in(points)

    # Adjust field extrapolation to maximum shift of secondary intermediate points
    shifted_points = points.shifted(0.5 * dt * vel_k1)
    shift = math.ceil(math.max(math.abs(shifted_points.center - points.center)))
    total_shift = shift
    velocity, smask = extp_sgrid(velocity, smask, int(shift))
    velocity *= bcs
    vel_k2 = velocity.sample_in(shifted_points)

    # Same for points of third and fourth component
    shifted_points = points.shifted(0.5 * dt * vel_k2)
    shift = math.ceil(math.max(math.abs(shifted_points.center - points.center))) - total_shift
    total_shift += shift
    velocity, smask = extp_sgrid(velocity, smask, int(shift))
    velocity *= bcs
    vel_k3 = velocity.sample_in(shifted_points)

    shifted_points = points.shifted(dt * vel_k3)
    shift = math.ceil(math.max(math.abs(shifted_points.center - points.center))) - total_shift
    velocity, _ = extp_sgrid(velocity, smask, int(shift))
    velocity *= bcs
    vel_k4 = velocity.sample_in(shifted_points)

    # --- Combine points with RK4 scheme ---
    vel = (1/6.) * (vel_k1 + 2 * (vel_k2 + vel_k3) + vel_k4)
    new_points = points.shifted(dt * vel)
    return PointCloud(new_points, field.values, field.extrapolation, add_overlapping=field._add_overlapping,
                      bounds=field.bounds, color=field.color)


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
    return PointCloud(new_points, field.values, field.extrapolation, add_overlapping=field._add_overlapping,
                      bounds=field.bounds, color=field.color)
