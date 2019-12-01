"""
Fluid utility functions.
"""
from numbers import Number

from phi import math
from .pressuresolver.base import FluidDomain
from .pressuresolver.sparse import SparseCG
from .field import CenteredGrid, StaggeredGrid, union_mask
from .material import OPEN, Material
from .domain import Domain


def _is_div_free(velocity, is_div_free):
    assert is_div_free in (True, False, None)
    if isinstance(is_div_free, bool):
        return is_div_free
    if isinstance(velocity, Number):
        return True
    return False


def solve_pressure(divergence, fluiddomain, pressure_solver=None):
    """
    Computes the pressure from the given velocity or velocity divergence using the specified solver.
    :param divergence: CenteredGrid
    :param fluiddomain: FluidDomain instance
    :param pressure_solver: PressureSolver to use, None for default
    :return: scalar tensor or CenteredGrid, depending on the type of divergence
    """
    assert isinstance(divergence, CenteredGrid)

    if pressure_solver is None:
        pressure_solver = SparseCG()

    pressure, iteration = pressure_solver.solve(divergence.data, fluiddomain, pressure_guess=None)

    if isinstance(divergence, CenteredGrid):
        pressure = CenteredGrid('pressure', pressure, divergence.box)

    return pressure, iteration


def divergence_free(velocity, domain=None, obstacles=(), pressure_solver=None):
    """
Projects the given velocity field by solving for and subtracting the pressure.
    :param velocity: StaggeredGrid
    :param domain: Domain matching the velocity field, used for boundary conditions
    :param obstacles: list of Obstacles
    :param pressure_solver: PressureSolver. Uses default solver if none provided.
    :return: divergence-free velocity as StaggeredGrid
    """
    assert isinstance(velocity, StaggeredGrid)
    if domain is None:
        domain = Domain(velocity.resolution, OPEN)
    obstacle_mask = union_mask([obstacle.geometry for obstacle in obstacles])
    if obstacle_mask is not None:
        obstacle_grid = obstacle_mask.at(velocity.center_points, collapse_dimensions=False)
        active_mask = 1 - obstacle_grid
    else:
        active_mask = math.ones(domain.centered_shape(name='active'))
    accessible_mask = active_mask.copied_with(extrapolation=Material.accessible_extrapolation_mode(domain.boundaries))
    fluiddomain = FluidDomain(domain, active=active_mask, accessible=accessible_mask)

    velocity = fluiddomain.with_hard_boundary_conditions(velocity)
    divergence_field = velocity.divergence(physical_units=False)
    pressure, _ = solve_pressure(divergence_field, fluiddomain, pressure_solver=pressure_solver)
    pressure *= velocity.dx[0]
    gradp = StaggeredGrid.gradient(pressure)
    velocity -= fluiddomain.with_hard_boundary_conditions(gradp)
    return velocity
