"""
Container for different advection schemes for grids and particles.

Esamples:

* semi_lagrangian (grid)
* mac_cormack (grid)
* runge_kutta_4 (particle)
"""

from phi import math
from phi.field import SampledField, ConstantField, StaggeredGrid, CenteredGrid, Grid, Field, PointCloud
from phi.field._field_math import GridType


def advect(field: Field, velocity: Field, dt):
    """
    Advect `field` along the `velocity` vectors using the default advection method.

    Args:
      field: any built-in Field
      velocity: any Field
      dt: time increment
      field: Field: 
      velocity: Field: 

    Returns:
      Advected field of same type as `field`

    """
    if isinstance(field, PointCloud):
        if isinstance(velocity, PointCloud) and velocity.elements == field.elements:
            return points(field, velocity, dt)
        return runge_kutta_4(field, velocity, dt=dt)
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
    return field._with(interpolated)


def mac_cormack(field: CenteredGrid, velocity: Field, dt, correction_strength=1.0) -> CenteredGrid:
    """
    MacCormack advection uses a forward and backward lookup to determine the first-order error of semi-Lagrangian advection.
    It then uses that error estimate to correct the field values.
    To avoid overshoots, the resulting value is bounded by the neighbouring grid cells of the backward lookup.

    Args:
      correction_strength: the estimated error is multiplied by this factor before being applied. The case correction_strength=0 equals semi-lagrangian advection. Set lower than 1.0 to avoid oscillations. (Default value = 1.0)
      field: Field to be advected
      velocity: vector field, need not be compatible with `field`.
      dt: time increment
      field: CenteredGrid: 
      velocity: Field: 

    Returns:
      Field compatible with input field

    """
    x0 = field.points
    v = velocity.sample_in(field.elements)
    x_bwd = x0 - v * dt
    x_fwd = x0 + v * dt
    field_semi_la = field._with(field.sample_at(x_bwd.values, reduce_channels='not yet implemented'))  # semi-Lagrangian advection
    field_inv_semi_la = field._with(field_semi_la.sample_at(x_fwd.values, reduce_channels='not yet implemented'))  # inverse semi-Lagrangian advection
    new_field = field_semi_la + correction_strength * 0.5 * (field - field_inv_semi_la)
    field_clamped = math.clip(new_field, *field.general_sample_at(x_bwd.values, 'minmax'))  # Address overshoots
    return field_clamped


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
