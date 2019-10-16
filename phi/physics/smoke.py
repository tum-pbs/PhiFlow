from .domain import *
from phi.field import *
from .fluid import *
from phi.solver.base import *
from phi.math.initializers import _is_python_shape, zeros, np


def initialize_field(value, shape, dtype=np.float32):
    if isinstance(value, (int, float)):
        const0 = zeros(shape, dtype=dtype)
        return const0 + value
    elif callable(value):
        return value(shape, dtype=dtype)
    if isinstance(shape, struct.Struct):
        if type(shape) == type(value):
            zipped = struct.zip([value, shape], leaf_condition=_is_python_shape)
            return struct.map(lambda val, sh: initialize_field(val, sh), zipped)
        else:
            return type(shape)(value)
    else:
        return value


class Smoke(State):
    __struct__ = State.__struct__.extend(('_density', '_velocity'),
                            ('_domain', '_gravity', '_buoyancy_factor', '_conserve_density'))

    def __init__(self, domain,
                 density=0.0, velocity=0,
                 gravity=-9.81, buoyancy_factor=0.1, conserve_density=False,
                 batch_size=None):
        State.__init__(self, tags=('smoke', 'velocityfield'), batch_size=batch_size)
        self._domain = domain
        self._density = initialize_field(density, self.domain.shape(1, self._batch_size, 'density'))
        self._velocity = initialize_field(velocity, self.domain.staggered_shape(self._batch_size, 'velocity'))
        self._gravity = gravity
        self._buoyancy_factor = buoyancy_factor
        self._conserve_density = conserve_density
        self.domaincache = None
        self._last_pressure = None
        self._last_pressure_iterations = None

    def default_physics(self):
        return SMOKE

    def copied_with(self, **kwargs):
        if 'density' in kwargs:
            kwargs['density'] = initialize_field(kwargs['density'], self.domain.shape(1, self._batch_size, 'density'))
        if 'velocity' in kwargs:
            kwargs['velocity'] = initialize_field(kwargs['velocity'], self.domain.staggered_shape(self._batch_size, 'velocity'))
        return State.copied_with(self, **kwargs)

    @property
    def density(self):
        return self._density

    @property
    def velocity(self):
        return self._velocity

    @property
    def domain(self):
        return self._domain

    @property
    def rank(self):
        return self.domain.rank

    @property
    def staggered_shape(self):
        return self.domain.staggered_shape(self._batch_size)

    @property
    def centered_shape(self):
        return self.domain.shape(1, self._batch_size)

    @property
    def gravity(self):
        return self._gravity

    @property
    def buoyancy_factor(self):
        return self._buoyancy_factor

    @property
    def conserve_density(self):
        return self._conserve_density

    @property
    def last_pressure(self):
        return self._last_pressure

    @property
    def last_pressure_iterations(self):
        return self._last_pressure_iterations

    def __repr__(self):
        return "Smoke[density: %s, velocity: %s]" % (self.density, self.velocity)


class SmokePhysics(Physics):

    def __init__(self, pressure_solver=None, make_input_divfree=False, make_output_divfree=True):
        Physics.__init__(self, dependencies={'obstacles': ['obstacle']},
                         blocking_dependencies={'density_effects': 'density_effect', 'velocity_effects': 'velocity_effect'})
        self.pressure_solver = pressure_solver
        self.make_input_divfree = make_input_divfree
        self.make_output_divfree = make_output_divfree

    def step(self, smoke, dt=1.0, obstacles=(), density_effects=(), velocity_effects=(), **dependent_states):
        assert len(dependent_states) == 0
        velocity = smoke.velocity
        density = smoke.density
        obstacle_mask = union([obstacle.geometry for obstacle in obstacles])
        if self.make_input_divfree:
            velocity = divergence_free(velocity, smoke.domain, obstacle_mask, pressure_solver=self.pressure_solver)
        # --- Advection ---
        density = advect.look_back(density, velocity, dt=dt)
        velocity = advect.look_back(velocity, velocity, dt=dt)
        # --- Density effects ---
        for effect in density_effects:
            density = effect_applied(effect, density, dt)
        # --- velocity effects
        for effect in velocity_effects:
            velocity = effect_applied(effect, velocity, dt)
        velocity += buoyancy(smoke.density, smoke.gravity, smoke.buoyancy_factor) * dt
        if self.make_output_divfree:
            velocity = divergence_free(velocity, smoke.domain, obstacle_mask, pressure_solver=self.pressure_solver)
        return smoke.copied_with(density=density, velocity=velocity, age=smoke.age + dt)


SMOKE = SmokePhysics()


def buoyancy(density, gravity, buoyancy_factor):
    if isinstance(gravity, (int, float)):
        gravity = np.array([gravity] + ([0] * (density.rank - 1)))
    result = StaggeredGrid.from_scalar(density, -gravity * buoyancy_factor)
    return result

