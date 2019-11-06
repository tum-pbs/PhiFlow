import numpy as np

from phi.physics import StateDependency
from .fluid import *


class Smoke(DomainState):

    def __init__(self, domain, density=0.0, velocity=0, buoyancy_factor=0.1, tags=('smoke', 'velocityfield'), **kwargs):
        DomainState.__init__(**struct.kwargs(locals()))

    def default_physics(self):
        return SMOKE

    @struct.attr(default=0)
    def density(self, density):
        return self.centered_grid('density', density)

    @struct.attr(default=0)
    def velocity(self, velocity):
        return self.staggered_grid('velocity', velocity)

    @struct.prop(default=0.1)
    def buoyancy_factor(self, fac): return fac

    def __repr__(self):
        return "Smoke[density: %s, velocity: %s]" % (self.density, self.velocity)


class SmokePhysics(Physics):

    def __init__(self, pressure_solver=None, make_input_divfree=False, make_output_divfree=True):
        Physics.__init__(self, [StateDependency('obstacles', 'obstacle'),
                                StateDependency('gravity', 'gravity'),
                                StateDependency('density_effects', 'density_effect', blocking=True),
                                StateDependency('velocity_effects', 'velocity_effect', blocking=True)])
        self.pressure_solver = pressure_solver
        self.make_input_divfree = make_input_divfree
        self.make_output_divfree = make_output_divfree

    def step(self, smoke, dt=1.0, obstacles=(), gravity=(), density_effects=(), velocity_effects=(), **dependent_states):
        assert len(dependent_states) == 0
        gravity = gravity_tensor(sum(gravity), smoke.rank)
        velocity = smoke.velocity
        density = smoke.density
        obstacle_mask = union_mask([obstacle.geometry for obstacle in obstacles])
        if self.make_input_divfree:
            velocity = divergence_free(velocity, smoke.domain, obstacle_mask, pressure_solver=self.pressure_solver)
        # --- Advection ---
        density = advect.semi_lagrangian(density, velocity, dt=dt)
        velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)
        # --- Density effects ---
        for effect in density_effects:
            density = effect_applied(effect, density, dt)
        # --- velocity effects
        for effect in velocity_effects:
            velocity = effect_applied(effect, velocity, dt)
        velocity += buoyancy(smoke.density, gravity, smoke.buoyancy_factor) * dt
        if self.make_output_divfree:
            velocity = divergence_free(velocity, smoke.domain, obstacle_mask, pressure_solver=self.pressure_solver)
        return smoke.copied_with(density=density, velocity=velocity, age=smoke.age + dt)


SMOKE = SmokePhysics()


def buoyancy(density, gravity, buoyancy_factor):
    if isinstance(gravity, (int, float)):
        gravity = np.array([gravity] + ([0] * (density.rank - 1)))
    result = StaggeredGrid.from_scalar(density, -gravity * buoyancy_factor)
    return result
