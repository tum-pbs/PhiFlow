"""
Definition of Fluid, IncompressibleFlow as well as fluid-related functions.
"""
from functools import partial

import numpy as np

from phi import math, struct, field
from phi.field import GeometryMask, AngularVelocity, CenteredGrid, Grid, divergence
from phi.geom import union, GridCell
from . import _advect
from ._physics import Physics, StateDependency, State
from ._effect import Gravity, effect_applied, gravity_tensor
from ._boundaries import Domain, Material, Obstacle


def build_masks(domain: Domain, obstacles=()):
    obstacle_mask = union([obstacle.geometry for obstacle in obstacles])
    active_mask = 1 - obstacle_mask.sample_at(GridCell(velocity.resolution, velocity.box).center)
    active_extrapolation = math.extrapolation.PERIODIC if velocity.extrapolation == math.extrapolation.PERIODIC else math.extrapolation.ZERO
    active_mask = CenteredGrid(active_mask, domain.box, active_extrapolation)
    active_mask = domain.grid(active_mask, Material.active_extrapolation)
    accessible_extrapolation = math.extrapolation.ONE if velocity.extrapolation in (math.extrapolation.PERIODIC, math.extrapolation.BOUNDARY) else math.extrapolation.ZERO
    accessible_mask = CenteredGrid(active_mask.values, active_mask.box, accessible_extrapolation)
    return active_mask, accessible_mask


def layer_obstacle_velocities(velocity: Grid, *obstacles: Obstacle):
    for obstacle in obstacles:
        if not obstacle.is_stationary:
            obs_mask = GeometryMask(obstacle.geometry).at(velocity)
            angular_velocity = AngularVelocity(location=obstacle.geometry.center, strength=obstacle.angular_velocity, falloff=None).at(velocity)
            obs_vel = (angular_velocity + obstacle.velocity).at(velocity)
            velocity = (1 - obs_mask) * velocity + obs_mask * obs_vel
    return velocity


def solve_pressure(velocity, active_mask, accessible_mask):
    divergence_field = field.divergence(velocity)
    pressure_extrapolation = velocity.extrapolation
    pressure_guess = CenteredGrid.sample(0, velocity.resolution, velocity.box, pressure_extrapolation)
    laplace_fun = partial(masked_laplace, active=active_mask, accessible=accessible_mask)
    converged, pressure, iterations = field.conjugate_gradient(laplace_fun, divergence_field, pressure_guess, relative_tolerance, absolute_tolerance, max_iterations, gradient)
    if not math.all(converged):
        raise AssertionError('pressure solve did not converge after %d iterations' % (iterations,))
    return pressure


def divergence_free_obstacles(velocity: Grid, obstacles=(), relative_tolerance: float = 1e-5, absolute_tolerance: float = 0.0, max_iterations: int = 1000, return_info=False):
    """
Projects the given velocity field by solving for and subtracting the pressure.
    :param return_info: if True, returns a dict holding information about the solve as a second object
    :param velocity: StaggeredGrid
    :param obstacles: list of Obstacles
    :return: divergence-free velocity as StaggeredGrid
    """
    active_mask, accessible_mask = build_masks(velocity)
    hard_bcs = field.stagger(accessible_mask, math.minimum)
    velocity *= hard_bcs
    velocity = layer_obstacle_velocities(velocity, obstacles)
    pressure = solve_pressure()
    gradp = field.staggered_gradient(pressure)
    gradp *= hard_bcs
    velocity -= gradp
    return velocity if not return_info else (velocity, {'pressure': pressure, 'iterations': iterations, 'divergence': divergence_field})


def masked_laplace(pressure: CenteredGrid, active: CenteredGrid, accessible: CenteredGrid) -> CenteredGrid:
    """
    Compute the laplace of a pressure-like field in the presence of obstacles.

    :param pressure: input field
    :param active: Scalar field encoding active cells as ones and inactive (open/obstacle) as zero.
        Active cells are those for which physical constants_dict such as pressure or velocity are calculated.
    :param accessible: Scalar field encoding cells that are accessible, i.e. not solid, as ones and obstacles as zero.
    :return: laplace of pressure given the boundary conditions
    """
    # TODO active * pressure has extrapolation=0
    left_act_pr, right_act_pr = field.shift(active * pressure, (-1, 1), 'vector')
    left_access, right_access = field.shift(accessible, (-1, 1), 'vector')
    left_right = (left_act_pr + right_act_pr) * active
    center = (left_access + right_access) * pressure
    result = (left_right - center) / pressure.dx ** 2
    result = math.sum(result.values, axis='vector')
    return CenteredGrid(result, pressure.box, pressure.extrapolation.gradient().gradient())


@struct.definition()
class Fluid(State):
    """
    A Fluid state consists of a density field (centered grid) and a velocity field (staggered grid).
    """

    def __init__(self, domain, density=0.0, velocity=0.0, buoyancy_factor=0.0, tags=('fluid', 'velocityfield', 'velocity'), name='fluid', **kwargs):
        State.__init__(self, **struct.kwargs(locals()))

    def default_physics(self):
        return IncompressibleFlow()

    @struct.constant()
    def domain(self, domain):
        return domain

    @struct.variable(default=0, dependencies='domain')
    def density(self, density):
        """
The marker density is stored in a CenteredGrid with dimensions matching the domain.
It describes the number of particles per physical volume.
        """
        return self.centered_grid('density', density)

    @struct.variable(default=0, dependencies='domain')
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

    @struct.variable(default={}, holds_data=False)
    def solve_info(self, solve_info):
        return dict(solve_info)

    def __repr__(self):
        return "Fluid[density: %s, velocity: %s]" % (self.density, self.velocity)


class IncompressibleFlow(Physics):
    """
Physics modelling the incompressible Navier-Stokes equations.
Supports buoyancy proportional to the marker density.
Supports obstacles, density effects, velocity effects, global gravity.
    """

    def __init__(self, make_input_divfree=False, make_output_divfree=True, conserve_density=True):
        Physics.__init__(self, [StateDependency('obstacles', 'obstacle', blocking=True),
                                StateDependency('gravity', 'gravity', single_state=True),
                                StateDependency('density_effects', 'density_effect', blocking=True),
                                StateDependency('velocity_effects', 'velocity_effect', blocking=True)])
        self.make_input_divfree = make_input_divfree
        self.make_output_divfree = make_output_divfree
        self.conserve_density = conserve_density

    def step(self, fluid, dt=1.0, obstacles=(), gravity=Gravity(), density_effects=(), velocity_effects=()):
        # pylint: disable-msg = arguments-differ
        gravity = gravity_tensor(gravity, fluid.rank)
        velocity = fluid.velocity
        density = fluid.density
        if self.make_input_divfree:
            velocity, solve_info = divergence_free(velocity, obstacles, return_info=True)
        # --- Advection ---
        density = _advect.semi_lagrangian(density, velocity, dt=dt)
        velocity = advected_velocity = _advect.semi_lagrangian(velocity, velocity, dt=dt)
        if self.conserve_density and np.all(Material.solid(fluid.domain.boundaries)):
            density = field.normalize(density, fluid.density)
        # --- Effects ---
        for effect in density_effects:
            density = effect_applied(effect, density, dt)
        for effect in velocity_effects:
            velocity = effect_applied(effect, velocity, dt)
        velocity += (density * -gravity * fluid.buoyancy_factor * dt).at(velocity)
        divergent_velocity = velocity
        # --- Pressure solve ---
        if self.make_output_divfree:
            velocity, solve_info = divergence_free(velocity, obstacles, return_info=True)
        solve_info['advected_velocity'] = advected_velocity
        solve_info['divergent_velocity'] = divergent_velocity
        return fluid.copied_with(density=density, velocity=velocity, age=fluid.age + dt, solve_info=solve_info)


