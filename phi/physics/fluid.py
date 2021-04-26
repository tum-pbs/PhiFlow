"""
Definition of Fluid, IncompressibleFlow as well as fluid-related functions.
"""

from phi import math, field
from phi.field import SoftGeometryMask, AngularVelocity, Grid, divergence, CenteredGrid, spatial_gradient, where, HardGeometryMask
from phi.geom import union
from ._boundaries import Domain


def make_incompressible(velocity: Grid,
                        domain: Domain,
                        obstacles: tuple or list = (),
                        solve_params: math.LinearSolve = math.LinearSolve(None, 1e-3),
                        pressure_guess: CenteredGrid = None):
    """
    Projects the given velocity field by solving for the pressure and subtracting its spatial_gradient.

    This method is similar to :func:`field.divergence_free()` but differs in how the boundary conditions are specified.

    Args:
      velocity: Vector field sampled on a grid
      domain: Used to specify boundary conditions
      obstacles: List of Obstacles to specify boundary conditions inside the domain (Default value = ())
      pressure_guess: Initial guess for the pressure solve
      solve_params: Parameters for the pressure solve

    Returns:
      velocity: divergence-free velocity of type `type(velocity)`
      pressure: solved pressure field, `CenteredGrid`
      iterations: Number of iterations required to solve for the pressure
      divergence: divergence field of input velocity, `CenteredGrid`

    """
    input_velocity = velocity
    velocity = apply_boundary_conditions(velocity, domain, obstacles).with_(extrapolation=domain.boundaries['near_vector_extrapolation'])
    div = divergence(velocity)
    if domain.boundaries['near_vector_extrapolation'] == math.extrapolation.BOUNDARY:
        div -= field.mean(div)
    active = domain.grid(HardGeometryMask(~union(*[obstacle.geometry for obstacle in obstacles])), extrapolation=domain.boundaries['active_extrapolation'])
    accessible = domain.grid(active, extrapolation=domain.boundaries['accessible_extrapolation'])
    hard_bcs = field.stagger(accessible, math.minimum, domain.boundaries['accessible_extrapolation'], type=type(velocity))

    # Solve pressure

    def laplace(p):
        grad = spatial_gradient(p, type(velocity))
        grad *= hard_bcs
        grad = grad.with_(extrapolation=domain.boundaries['near_vector_extrapolation'])
        div = divergence(grad)
        lap = where(active, div, p)
        return lap

    pressure_guess = pressure_guess if pressure_guess is not None else domain.scalar_grid(0)
    converged, pressure, iterations = field.solve(laplace, y=div, x0=pressure_guess, solve_params=solve_params, constants=[active, hard_bcs])
    if math.all_available(converged) and not math.all(converged):
        raise AssertionError(f"pressure solve did not converge after {iterations} iterations\nResult: {pressure.values}")
    # Subtract grad pressure
    gradp = field.spatial_gradient(pressure, type=type(velocity)) * hard_bcs
    velocity = (velocity - gradp).with_(extrapolation=input_velocity.extrapolation)
    return velocity, pressure, iterations, div


def apply_boundary_conditions(velocity: Grid, domain : Domain, obstacles: tuple or list):
    """
    Enforces velocities boundary conditions on a velocity grid.
    Cells inside obstacles will get their velocity from the obstacle movement.
    Cells outside far away will be unaffected.

    Args:
      velocity: centered or staggered velocity grid
      domain: simulation domain
      obstacles: sequence of Obstacles
      velocity: Grid:
      domain: Domain:
      obstacles: tuple or list:

    Returns:
      velocity of same type as `velocity`

    """
    # Mask hard boundaries and obstacle
    bcs = field.stagger(domain.scalar_grid(1, domain.boundaries['accessible_extrapolation']), math.minimum, domain.boundaries['accessible_extrapolation'], type=type(velocity))
    bcs *= 1 - (HardGeometryMask(union([obstacle.geometry for obstacle in obstacles])) >> bcs )
    velocity *= bcs
    # Add obstacle velocity to fluid
    for obstacle in obstacles:
        if not obstacle.is_stationary:
            obs_mask = SoftGeometryMask(obstacle.geometry, balance=1) >> velocity
            angular_velocity = AngularVelocity(location=obstacle.geometry.center, strength=obstacle.angular_velocity, falloff=None).at(velocity)
            obs_vel = angular_velocity + obstacle.velocity
            velocity = (1 - obs_mask) * velocity + obs_mask * obs_vel
    return velocity



