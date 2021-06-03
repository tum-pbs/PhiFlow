"""
Definition of Fluid, IncompressibleFlow as well as fluid-related functions.
"""
from typing import Tuple

from phi import math, field
from phi.field import SoftGeometryMask, AngularVelocity, Grid, divergence, spatial_gradient, where, HardGeometryMask, CenteredGrid
from phi.geom import union
from ._boundaries import Domain
from ..math._tensors import copy_with


def make_incompressible(velocity: Grid,
                        boundaries: dict or Domain,
                        obstacles: tuple or list = (),
                        solve=math.Solve('auto', 1e-5, 0, gradient_solve=math.Solve('auto', 1e-5, 1e-5))) -> Tuple[Grid, CenteredGrid]:
    """
    Projects the given velocity field by solving for the pressure and subtracting its spatial_gradient.
    
    This method is similar to :func:`field.divergence_free()` but differs in how the boundary conditions are specified.

    Args:
      velocity: Vector field sampled on a grid
      boundaries: Boundary conditions as `dict` or `Domain`.
      obstacles: List of Obstacles to specify boundary conditions inside the domain (Default value = ())
      solve: Parameters for the pressure solve as.

    Returns:
      velocity: divergence-free velocity of type `type(velocity)`
      pressure: solved pressure field, `CenteredGrid`
    """
    boundaries = boundaries.boundaries if isinstance(boundaries, Domain) else boundaries
    input_velocity = velocity
    active = CenteredGrid(HardGeometryMask(~union(*[obstacle.geometry for obstacle in obstacles])), resolution=velocity.resolution, bounds=velocity.bounds, extrapolation=boundaries['active'])
    accessible = active.with_(extrapolation=boundaries['accessible'])
    hard_bcs = field.stagger(accessible, math.minimum, boundaries['accessible'], type=type(velocity))
    v_bc_div = boundaries['vector'] * boundaries['accessible']
    velocity = apply_boundary_conditions(velocity, obstacles, boundaries).with_(extrapolation=v_bc_div)
    div = divergence(velocity) * active
    if boundaries['accessible'] == math.extrapolation.ZERO or boundaries['vector'] == math.extrapolation.PERIODIC:
        div = _balance_divergence(div, active)
        # math.assert_close(field.mean(div), 0, abs_tolerance=1e-6)

    # Solve pressure
    @math.jit_compile_linear
    def laplace(p):  # TODO when called during backward, the forward jit is already done but tracers are still referenced...
        # TODO active, hard_bcs are actually arguments
        grad = spatial_gradient(p, type(velocity))
        grad *= hard_bcs
        grad = grad.with_(extrapolation=v_bc_div)
        div = divergence(grad)
        lap = where(active, div, p)
        return lap

    if solve.x0 is None:
        solve = copy_with(solve, x0=CenteredGrid(0, resolution=div.resolution, bounds=div.bounds, extrapolation=boundaries['scalar']))
    pressure = field.solve_linear(laplace, y=div, solve=solve)
    if boundaries['accessible'] == math.extrapolation.ZERO or boundaries['vector'] == math.extrapolation.PERIODIC:
        def pressure_backward(_p, _p_, dp: CenteredGrid):
            # re-generate active mask because value might not be accessible from forward pass (e.g. Jax jit)
            active = CenteredGrid(HardGeometryMask(~union(*[obstacle.geometry for obstacle in obstacles])), resolution=dp.resolution, bounds=dp.bounds, extrapolation=boundaries['active'])
            return _balance_divergence(dp, active),
        pressure = math.custom_gradient(lambda p: p, pressure_backward)(pressure)
    # Subtract grad pressure
    gradp = field.spatial_gradient(pressure, type=type(velocity)) * hard_bcs
    velocity = (velocity - gradp).with_(extrapolation=input_velocity.extrapolation)
    return velocity, pressure


def _balance_divergence(div, active):
    return div - active * (field.mean(div) / field.mean(active))


def apply_boundary_conditions(velocity: Grid, obstacles: tuple or list, boundaries: dict):
    """
    Enforces velocities boundary conditions on a velocity grid.
    Cells inside obstacles will get their velocity from the obstacle movement.
    Cells outside far away will be unaffected.

    Args:
      velocity: centered or staggered velocity grid
      boundaries: simulation domain
      obstacles: sequence of Obstacles

    Returns:
      velocity of same type as `velocity`

    """
    # Mask hard boundaries and obstacle
    bcs = field.stagger(CenteredGrid(1, resolution=velocity.resolution, bounds=velocity.bounds, extrapolation=boundaries['accessible']), math.minimum, boundaries['accessible'], type=type(velocity))
    bcs *= 1 - (HardGeometryMask(union([obstacle.geometry for obstacle in obstacles])) >> bcs)
    velocity *= bcs
    # Add obstacle velocity to fluid
    for obstacle in obstacles:
        if not obstacle.is_stationary:
            obs_mask = SoftGeometryMask(obstacle.geometry, balance=1) >> velocity
            angular_velocity = AngularVelocity(location=obstacle.geometry.center, strength=obstacle.angular_velocity, falloff=None).at(velocity)
            obs_vel = angular_velocity + obstacle.velocity
            velocity = (1 - obs_mask) * velocity + obs_mask * obs_vel
    return velocity
