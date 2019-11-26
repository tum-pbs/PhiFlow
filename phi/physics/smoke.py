"""
This file defines Smoke states and SmokePhysics.
More general fluid functions are located in fluid.py
"""

import numpy as np

from phi import struct
from .physics import StateDependency, Physics
from .field import advect, StaggeredGrid
from .field.effect import Gravity, gravity_tensor, effect_applied
from .domain import DomainState
from .fluid import divergence_free


@struct.definition()
class Smoke(DomainState):
    """
    A Smoke state consists of a density field (centered grid) and a velocity field (staggered grid).
    """

    def __init__(self, domain, density=0.0, velocity=0, buoyancy_factor=0.1,
                 tags=('smoke', 'velocityfield'), name='smoke', **kwargs):
        DomainState.__init__(self, **struct.kwargs(locals()))

    def default_physics(self):
        """
        :return: SMOKE
        """
        return SMOKE

    @struct.attr(default=0, dependencies=DomainState.domain)
    def density(self, density):
        """
The smoke density is stored in a CenteredGrid with dimensions matching the domain.
It describes the number of smoke particles per volume.
        """
        return self.centered_grid('density', density)

    @struct.attr(default=0, dependencies=DomainState.domain)
    def velocity(self, velocity):
        """
The velocity is stored in a StaggeredGrid with dimensions matching the domain.
        """
        return self.staggered_grid('velocity', velocity)

    @struct.prop(default=0.1)
    def buoyancy_factor(self, fac):
        """
The default smoke physics applies buoyancy as an upward force.
This force is scaled with the buoyancy_factor (float).
        """
        return fac

    def __repr__(self):
        return "Smoke[density: %s, velocity: %s]" % (self.density, self.velocity)


class SmokePhysics(Physics):
    """
Default smoke physics modelling incompressible air flow with buoyancy proportional to the smoke density.
Supports obstacles, density effects, velocity effects, global gravity.
    """

    def __init__(self, pressure_solver=None, make_input_divfree=False, make_output_divfree=True):
        Physics.__init__(self, [StateDependency('obstacles', 'obstacle'),
                                StateDependency('gravity', 'gravity', single_state=True),
                                StateDependency('density_effects', 'density_effect', blocking=True),
                                StateDependency('velocity_effects', 'velocity_effect', blocking=True)])
        self.pressure_solver = pressure_solver
        self.make_input_divfree = make_input_divfree
        self.make_output_divfree = make_output_divfree

    def step(self, smoke, dt=1.0, obstacles=(), gravity=Gravity(), density_effects=(), velocity_effects=()):
        # pylint: disable-msg = arguments-differ
        gravity = gravity_tensor(gravity, smoke.rank)
        velocity = smoke.velocity
        density = smoke.density
        if self.make_input_divfree:
            velocity = divergence_free(velocity, smoke.domain, obstacles, pressure_solver=self.pressure_solver)
        # --- Advection ---
        density = advect.semi_lagrangian(density, velocity, dt=dt)
        velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)
        # --- Effects ---
        for effect in density_effects:
            density = effect_applied(effect, density, dt)
        for effect in velocity_effects:
            velocity = effect_applied(effect, velocity, dt)
        velocity += buoyancy(smoke.density, gravity, smoke.buoyancy_factor) * dt
        # --- Pressure solve ---
        if self.make_output_divfree:
            velocity = divergence_free(velocity, smoke.domain, obstacles, pressure_solver=self.pressure_solver)
        return smoke.copied_with(density=density, velocity=velocity, age=smoke.age + dt)


SMOKE = SmokePhysics()


def buoyancy(density, gravity, buoyancy_factor):
    """
Computes the buoyancy force proportional to the density.
    :param density: CenteredGrid
    :param gravity: vector or float
    :param buoyancy_factor: float
    :return: StaggeredGrid for the domain of the density
    """
    if isinstance(gravity, (int, float)):
        gravity = np.array([gravity] + ([0] * (density.rank - 1)))
    result = StaggeredGrid.from_scalar(density, -gravity * buoyancy_factor)
    return result
