"""
Definition of Fluid, IncompressibleFlow as well as fluid-related functions.
"""
from typing import Tuple

from phi import math, field
from phi.field import SoftGeometryMask, AngularVelocity, Grid, divergence, spatial_gradient, where, HardGeometryMask, CenteredGrid
from phi.geom import union
from ..math import extrapolation
from ..math._tensors import copy_with
from ..math.extrapolation import combine_sides


def make_incompressible(velocity: Grid,
                        obstacles: tuple or list = (),
                        solve=math.Solve('auto', 1e-5, 0, gradient_solve=math.Solve('auto', 1e-5, 1e-5))) -> Tuple[Grid, CenteredGrid]:
    """
    Projects the given velocity field by solving for the pressure and subtracting its spatial_gradient.
    
    This method is similar to :func:`field.divergence_free()` but differs in how the boundary conditions are specified.

    Args:
        velocity: Vector field sampled on a grid
        obstacles: List of Obstacles to specify boundary conditions inside the domain (Default value = ())
        solve: Parameters for the pressure solve as.

    Returns:
        velocity: divergence-free velocity of type `type(velocity)`
        pressure: solved pressure field, `CenteredGrid`
    """
    assert isinstance(obstacles, (tuple, list)), f"obstacles must be a tuple or list but got {type(obstacles)}"
    input_velocity = velocity
    accessible_extrapolation = _accessible_extrapolation(input_velocity.extrapolation)
    active = CenteredGrid(HardGeometryMask(~union(*[obstacle.geometry for obstacle in obstacles])), resolution=velocity.resolution, bounds=velocity.bounds, extrapolation=extrapolation.NONE)
    accessible = active.with_extrapolation(accessible_extrapolation)
    hard_bcs = field.stagger(accessible, math.minimum, input_velocity.extrapolation, type=type(velocity))
    velocity = apply_boundary_conditions(velocity, obstacles)
    div = divergence(velocity) * active
    if input_velocity.extrapolation in (math.extrapolation.ZERO, math.extrapolation.PERIODIC):
        div = _balance_divergence(div, active)
    if solve.x0 is None:
        pressure_extrapolation = _pressure_extrapolation(input_velocity.extrapolation)
        solve = copy_with(solve, x0=CenteredGrid(0, resolution=div.resolution, bounds=div.bounds, extrapolation=pressure_extrapolation))

    pressure = math.solve_linear(masked_laplace, f_args=[hard_bcs, active], y=div, solve=solve)

    # if input_velocity.extrapolation in (math.extrapolation.ZERO, math.extrapolation.PERIODIC):
    #     def pressure_backward(_p, _p_, dp: CenteredGrid):
    #         # re-generate active mask because value might not be accessible from forward pass (e.g. Jax jit)
    #         active = CenteredGrid(HardGeometryMask(~union(*[obstacle.geometry for obstacle in obstacles])), resolution=dp.resolution, bounds=dp.bounds, extrapolation=extrapolation.NONE)
    #         return _balance_divergence(dp, active),
    #     pressure = math.custom_gradient(lambda p: p, pressure_backward)(pressure)

    grad_pressure = field.spatial_gradient(pressure, input_velocity.extrapolation, type=type(velocity)) * hard_bcs
    velocity = velocity - grad_pressure
    return velocity, pressure


@math.jit_compile_linear
def masked_laplace(pressure: CenteredGrid, hard_bcs: Grid, active: CenteredGrid):
    grad = spatial_gradient(pressure, hard_bcs.extrapolation, type=type(hard_bcs))
    grad *= hard_bcs
    div = divergence(grad)
    lap = where(active, div, pressure)
    return lap


def _balance_divergence(div, active):
    return div - active * (field.mean(div) / field.mean(active))


def apply_boundary_conditions(velocity: Grid, obstacles: tuple or list):
    """
    Enforces velocities boundary conditions on a velocity grid.
    Cells inside obstacles will get their velocity from the obstacle movement.
    Cells outside far away will be unaffected.

    Args:
      velocity: Velocity `Grid`.
      obstacles: Obstacles as `tuple` or `list`

    Returns:
        Velocity of same type as `velocity`
    """
    # velocity = field.bake_extrapolation(velocity)  # TODO we should bake only for divergence but keep correct extrapolation for velocity. However, obstacles should override extrapolation.
    for obstacle in obstacles:
        obs_mask = SoftGeometryMask(obstacle.geometry, balance=1) @ velocity
        if obstacle.is_stationary:
            velocity = (1 - obs_mask) * velocity
        else:
            angular_velocity = AngularVelocity(location=obstacle.geometry.center, strength=obstacle.angular_velocity, falloff=None) @ velocity
            velocity = (1 - obs_mask) * velocity + obs_mask * (angular_velocity + obstacle.velocity)
    return velocity


def _pressure_extrapolation(vext: math.Extrapolation):
    if vext == extrapolation.PERIODIC:
        return extrapolation.PERIODIC
    elif vext == extrapolation.BOUNDARY:
        return extrapolation.ZERO
    elif isinstance(vext, extrapolation.ConstantExtrapolation):
        return extrapolation.BOUNDARY
    elif isinstance(vext, extrapolation._MixedExtrapolation):
        return combine_sides(**{dim: (_pressure_extrapolation(lo), _pressure_extrapolation(hi)) for dim, (lo, hi) in vext.ext.items()})
    else:
        raise ValueError(f"Unsupported extrapolation: {type(vext)}")


def _accessible_extrapolation(vext: math.Extrapolation):
    if vext == extrapolation.PERIODIC:
        return extrapolation.PERIODIC
    elif vext == extrapolation.BOUNDARY:
        return extrapolation.ONE
    elif isinstance(vext, extrapolation.ConstantExtrapolation):
        return extrapolation.ZERO
    elif isinstance(vext, extrapolation._MixedExtrapolation):
        return combine_sides(**{dim: (_accessible_extrapolation(lo), _accessible_extrapolation(hi)) for dim, (lo, hi) in vext.ext.items()})
    else:
        raise ValueError(f"Unsupported extrapolation: {type(vext)}")
