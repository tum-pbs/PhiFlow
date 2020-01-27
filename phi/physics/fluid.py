"""
Definition of Fluid, IncompressibleFlow as well as fluid-related functions.
"""
from numbers import Number

import numpy as np
from phi import math, struct

from .domain import Domain, DomainState
from .field import CenteredGrid, StaggeredGrid, advect, union_mask
from .field.effect import Gravity, effect_applied, gravity_tensor
from .material import OPEN, Material
from .physics import Physics, StateDependency
from .pressuresolver.solver_api import FluidDomain
from .pressuresolver.sparse import SparseCG


@struct.definition()
class Fluid(DomainState):
    """
    A Fluid state consists of a density field (centered grid) and a velocity field (staggered grid).
    """

    def __init__(self, domain, density=0.0, velocity=0.0, buoyancy_factor=0.0, tags=('fluid', 'velocityfield'), name='fluid', **kwargs):
        DomainState.__init__(self, **struct.kwargs(locals()))
        self.solve_info = {}

    def default_physics(self): return INCOMPRESSIBLE_FLOW

    @struct.variable(default=0, dependencies=DomainState.domain)
    def density(self, density):
        """
The marker density is stored in a CenteredGrid with dimensions matching the domain.
It describes the number of particles per physical volume.
        """
        return self.centered_grid('density', density)

    @struct.variable(default=0, dependencies=DomainState.domain)
    def velocity(self, velocity):
        """
The velocity is stored in a StaggeredGrid with dimensions matching the domain.
        """
        return self.staggered_grid('velocity', velocity)

    @struct.constant(default=0.0)
    def buoyancy_factor(self, fac):
        """
The default fluid physics can apply Boussinesq buoyancy as an upward force, proportional to the density.
This force is scaled with the buoyancy_factor (float).
        """
        return fac

    def __repr__(self):
        return "Fluid[density: %s, velocity: %s]" % (self.density, self.velocity)


class IncompressibleFlow(Physics):
    """
Physics modelling the incompressible Navier-Stokes equations.
Supports buoyancy proportional to the marker density.
Supports obstacles, density effects, velocity effects, global gravity.
    """

    def __init__(self, pressure_solver=None, make_input_divfree=False, make_output_divfree=True, conserve_density=True):
        Physics.__init__(self, [StateDependency('obstacles', 'obstacle'),
                                StateDependency('gravity', 'gravity', single_state=True),
                                StateDependency('density_effects', 'density_effect', blocking=True),
                                StateDependency('velocity_effects', 'velocity_effect', blocking=True)])
        self.pressure_solver = pressure_solver
        self.make_input_divfree = make_input_divfree
        self.make_output_divfree = make_output_divfree
        self.conserve_density = conserve_density

    def step(self, fluid, dt=1.0, obstacles=(), gravity=Gravity(), density_effects=(), velocity_effects=()):
        # pylint: disable-msg = arguments-differ
        gravity = gravity_tensor(gravity, fluid.rank)
        velocity = fluid.velocity
        density = fluid.density
        if self.make_input_divfree:
            velocity, fluid.solve_info = divergence_free(velocity, fluid.domain, obstacles, pressure_solver=self.pressure_solver, return_info=True)
        # --- Advection ---
        density = advect.semi_lagrangian(density, velocity, dt=dt)
        velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)
        if self.conserve_density and np.all(Material.solid(fluid.domain.boundaries)):
            density = density.normalized(fluid.density)
        # --- Effects ---
        for effect in density_effects:
            density = effect_applied(effect, density, dt)
        for effect in velocity_effects:
            velocity = effect_applied(effect, velocity, dt)
        velocity += buoyancy(fluid.density, gravity, fluid.buoyancy_factor) * dt
        # --- Pressure solve ---
        if self.make_output_divfree:
            velocity, fluid.solve_info = divergence_free(velocity, fluid.domain, obstacles, pressure_solver=self.pressure_solver, return_info=True)
        return fluid.copied_with(density=density, velocity=velocity, age=fluid.age + dt)


INCOMPRESSIBLE_FLOW = IncompressibleFlow()


def buoyancy(density, gravity, buoyancy_factor):
    """
Computes the buoyancy force proportional to the density.
    :param density: CenteredGrid
    :param gravity: vector or float
    :param buoyancy_factor: float
    :return: StaggeredGrid for the domain of the density
    """
    if isinstance(gravity, (int, float)):
        gravity = math.to_float(math.as_tensor([gravity] + ([0] * (density.rank - 1))))
    result = StaggeredGrid.from_scalar(density, -gravity * buoyancy_factor)
    return result


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
    :return: pressure field, iteration count
    :rtype: CenteredGrid, int
    """
    assert isinstance(divergence, CenteredGrid)
    if pressure_solver is None:
        pressure_solver = SparseCG()
    pressure, iteration = pressure_solver.solve(divergence.data, fluiddomain, pressure_guess=None)
    if isinstance(divergence, CenteredGrid):
        pressure = CenteredGrid(pressure, divergence.box, name='pressure')
    return pressure, iteration


def divergence_free(velocity, domain=None, obstacles=(), pressure_solver=None, return_info=False):
    """
Projects the given velocity field by solving for and subtracting the pressure.
    :param return_info: if True, returns a dict holding information about the solve as a second object
    :param velocity: StaggeredGrid
    :param domain: Domain matching the velocity field, used for boundary conditions
    :param obstacles: list of Obstacles
    :param pressure_solver: PressureSolver. Uses default solver if none provided.
    :return: divergence-free velocity as StaggeredGrid
    """
    assert isinstance(velocity, StaggeredGrid)
    # --- Set up FluidDomain ---
    if domain is None:
        domain = Domain(velocity.resolution, OPEN)
    obstacle_mask = union_mask([obstacle.geometry for obstacle in obstacles])
    if obstacle_mask is not None:
        obstacle_grid = obstacle_mask.at(velocity.center_points, collapse_dimensions=False).copied_with(extrapolation='constant')
        active_mask = 1 - obstacle_grid
    else:
        active_mask = math.ones(domain.centered_shape(name='active', extrapolation='constant'))
    accessible_mask = active_mask.copied_with(extrapolation=Material.accessible_extrapolation_mode(domain.boundaries))
    fluiddomain = FluidDomain(domain, active=active_mask, accessible=accessible_mask)
    # --- Boundary Conditions, Pressure Solve ---
    velocity = fluiddomain.with_hard_boundary_conditions(velocity)
    divergence_field = velocity.divergence(physical_units=False)
    pressure, iterations = solve_pressure(divergence_field, fluiddomain, pressure_solver=pressure_solver)
    pressure *= velocity.dx[0]
    gradp = StaggeredGrid.gradient(pressure)
    velocity -= fluiddomain.with_hard_boundary_conditions(gradp)
    return velocity if not return_info else (velocity, {'pressure': pressure, 'iterations': iterations})
