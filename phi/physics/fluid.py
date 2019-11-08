from phi.physics.pressuresolver.base import *

from .domain import *
from phi.physics.field.effect import *


def _is_div_free(velocity, is_div_free):
    assert is_div_free in (True, False, None)
    if isinstance(is_div_free, bool):
        return is_div_free
    if isinstance(velocity, Number):
        return True
    return False


def solve_pressure(divergence, fluiddomain, pressure_solver=None):
    """
    Calculates the pressure from the given velocity or velocity divergence using the specified solver.
    
    :param divergence: CenteredGrid
    :param fluiddomain: FluidDomain instance
    :param pressure_solver: PressureSolver to use, None for default
    :return: scalar tensor or CenteredGrid, depending on the type of divergence
    """
    assert isinstance(divergence, CenteredGrid)

    if pressure_solver is None:
        from phi.physics.pressuresolver.sparse import SparseCG
        pressure_solver = SparseCG()

    pressure, iteration = pressure_solver.solve(divergence.data, fluiddomain, pressure_guess=None)

    if isinstance(divergence, CenteredGrid):
        pressure = CenteredGrid('pressure', divergence.box, pressure)

    return pressure, iteration


def divergence_free(velocity, domain=None, obstacle_mask=None, pressure_solver=None):
    assert isinstance(velocity, StaggeredGrid)
    if domain is None:
        domain = Domain(velocity.resolution, OPEN)
    if obstacle_mask is not None:
        obstacle_grid = obstacle_mask.at(velocity.center_points, collapse_dimensions=False).data
        active_mask = 1 - obstacle_grid
    else:
        active_mask = math.ones(domain.centered_shape(name='active')).data
    fluiddomain = FluidDomain(domain, active=active_mask, accessible=active_mask)

    velocity = fluiddomain.with_hard_boundary_conditions(velocity)
    divergence_field = velocity.divergence(physical_units=False)
    pressure, iteration = solve_pressure(divergence_field, fluiddomain, pressure_solver=pressure_solver)
    pressure *= velocity.dx[0]
    gradp = StaggeredGrid.gradient(pressure)
    velocity -= fluiddomain.with_hard_boundary_conditions(gradp)
    return velocity
