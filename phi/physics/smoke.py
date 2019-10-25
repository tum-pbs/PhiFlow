from .domain import *
from phi.field import *
from .fluid import *
from phi.solver.base import *
from phi.math.initializers import _is_python_shape, zeros, np


class Smoke(DomainState):
    __struct__ = DomainState.__struct__.extend(('_density', '_velocity'),
                            ('_buoyancy_factor', '_conserve_density'))

    def __init__(self, domain,
                 density=0.0, velocity=0,
                 buoyancy_factor=0.1, conserve_density=False,
                 batch_size=None):
        DomainState.__init__(self, domain, tags=('smoke', 'velocityfield'), batch_size=batch_size)
        self._density = density
        self._velocity = velocity
        self._buoyancy_factor = buoyancy_factor
        self._conserve_density = conserve_density
        self.domaincache = None
        self.__validate__()

    def default_physics(self):
        return SMOKE

    def __validate_density__(self):
        self._density = self.centered_grid('density', self._density)

    def __validate_velocity__(self):
        self._velocity = self.staggered_grid('velocity', self._velocity)

    @property
    def density(self):
        return self._density

    @property
    def velocity(self):
        return self._velocity

    @property
    def buoyancy_factor(self):
        return self._buoyancy_factor

    @property
    def conserve_density(self):
        return self._conserve_density

    def __repr__(self):
        return "Smoke[density: %s, velocity: %s]" % (self.density, self.velocity)


class SmokePhysics(Physics):

    def __init__(self, pressure_solver=None, make_input_divfree=False, make_output_divfree=True):
        Physics.__init__(self, dependencies={'obstacles': ['obstacle'], 'gravity': 'gravity'},
                         blocking_dependencies={'density_effects': 'density_effect', 'velocity_effects': 'velocity_effect'})
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

