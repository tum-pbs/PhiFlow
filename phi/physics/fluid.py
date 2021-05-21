"""
Definition of Fluid, IncompressibleFlow as well as fluid-related functions.
"""
from typing import Tuple

from phi import math, field
from phi.field import GeometryMask, AngularVelocity, Grid, divergence, spatial_gradient, where, HardGeometryMask
from phi.geom import union
from ._boundaries import Domain
from ..math import SolveResult
from ..math._tensors import copy_with


def make_incompressible(velocity: Grid,
                        domain: Domain,
                        obstacles: tuple or list = (),
                        solve=math.Solve('CG-adaptive', 1e-5, 0)) -> Tuple[Grid, SolveResult]:
    """
    Projects the given velocity field by solving for the pressure and subtracting its spatial_gradient.
    
    This method is similar to :func:`field.divergence_free()` but differs in how the boundary conditions are specified.

    Args:
      velocity: Vector field sampled on a grid
      domain: Used to specify boundary conditions
      obstacles: List of Obstacles to specify boundary conditions inside the domain (Default value = ())
      pressure_guess: Initial guess for the pressure solve_linear
      solve: Parameters for the pressure solve_linear

    Returns:
      velocity: divergence-free velocity of type `type(velocity)`
      pressure: solved pressure field, `CenteredGrid`
      iterations: Number of iterations required to solve_linear for the pressure
      divergence: divergence field of input velocity, `CenteredGrid`
    """
    input_velocity = velocity
    active = domain.scalar_grid(HardGeometryMask(~union(*[obstacle.geometry for obstacle in obstacles])), extrapolation='active')
    accessible = domain.scalar_grid(active, extrapolation='accessible')
    hard_bcs = field.stagger(accessible, math.minimum, domain.boundaries['accessible'], type=type(velocity))
    v_bc_div = domain.boundaries['vector'] * domain.boundaries['accessible']
    velocity = layer_obstacle_velocities(velocity * hard_bcs, obstacles).with_(extrapolation=v_bc_div)
    div = divergence(velocity) * active
    if domain.boundaries['accessible'] == math.extrapolation.ZERO or domain.boundaries['vector'] == math.extrapolation.PERIODIC:
        div = _balance_divergence(div, active)
        # math.assert_close(field.mean(div), 0, abs_tolerance=1e-6)

    # Solve pressure
    @math.jit_compile_linear
    def laplace(p):
        grad = spatial_gradient(p, type(velocity))
        grad *= hard_bcs
        grad = grad.with_(extrapolation=v_bc_div)
        div = divergence(grad)
        lap = where(active, div, p)
        return lap

    active.values._expand()
    hard_bcs.values._expand()
    if not solve.x0:
        solve = copy_with(solve, x0=domain.scalar_grid(0))
    pressure = field.solve_linear(laplace, y=div, solve=solve)
    if domain.boundaries['accessible'] == math.extrapolation.ZERO or domain.boundaries['vector'] == math.extrapolation.PERIODIC:
        def pressure_backward(_p, _p_, dp):
            # re-generate active mask because value might not be accessible from forward pass (e.g. Jax jit)
            active = domain.scalar_grid(HardGeometryMask(~union(*[obstacle.geometry for obstacle in obstacles])), extrapolation='active')
            return _balance_divergence(div.with_(values=dp), active).values,
        remove_div_in_gradient = math.custom_gradient(lambda p: p, pressure_backward)
        pressure = pressure.with_(values=remove_div_in_gradient(pressure.values))
    # Subtract grad pressure
    gradp = field.spatial_gradient(pressure, type=type(velocity)) * hard_bcs
    velocity = (velocity - gradp).with_(extrapolation=input_velocity.extrapolation)
    return velocity, pressure


def _balance_divergence(div, active):
    return div - active * (field.mean(div) / field.mean(active))


def layer_obstacle_velocities(velocity: Grid, obstacles: tuple or list):
    """
    Enforces obstacle boundary conditions on a velocity grid.
    Cells inside obstacles will get their velocity from the obstacle movement.
    Cells outside will be unaffected.

    Args:
      velocity: centered or staggered velocity grid
      obstacles: sequence of Obstacles
      velocity: Grid: 
      obstacles: tuple or list: 

    Returns:
      velocity of same type as `velocity`

    """
    for obstacle in obstacles:
        if not obstacle.is_stationary:
            obs_mask = GeometryMask(obstacle.geometry)
            obs_mask = obs_mask.at(velocity)
            angular_velocity = AngularVelocity(location=obstacle.geometry.center, strength=obstacle.angular_velocity, falloff=None).at(velocity)
            obs_vel = angular_velocity + obstacle.velocity
            velocity = (1 - obs_mask) * velocity + obs_mask * obs_vel
    return velocity



