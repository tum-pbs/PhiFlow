from .domain import *
from phi.field import *
from .fluid import *
from phi.solver.base import *
from phi.math.initializers import _is_python_shape, zeros, np


def initialize_field(value, shape, dtype=np.float32):
    if isinstance(value, (int, float)):
        return zeros(shape, dtype=dtype) + value
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
        self._density = initialize_field(density, self.domain.shape(1, self._batch_size))
        self._velocity = initialize_field(velocity, self.domain.staggered_shape(self._batch_size))
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
            kwargs['density'] = initialize_field(kwargs['density'], self.domain.shape(1, self._batch_size))
        if 'velocity' in kwargs:
            kwargs['velocity'] = initialize_field(kwargs['velocity'], self.domain.staggered_shape(self._batch_size))
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
        if self.make_input_divfree:
            velocity = divergence_free(velocity, self.pressure_solver)
        # --- Advection ---
        density = velocity.advect(density, dt=dt)
        velocity = velocity.advect(velocity, dt=dt)
        # --- Density effects ---
        for effect in density_effects:
            density = effect.apply_grid(density, smoke.domain, staggered=False, dt=dt)
        # --- velocity effects
        for effect in velocity_effects:
            velocity = effect.apply_grid(velocity, smoke.domain, staggered=True, dt=dt)
        velocity += dt * buoyancy(smoke.density, smoke.gravity, smoke.buoyancy_factor)
        if self.make_output_divfree:
            velocity = divergence_free(velocity, self.pressure_solver)
        return smoke.copied_with(density=density, velocity=velocity, age=smoke.age + dt)


SMOKE = SmokePhysics()


def buoyancy(density, gravity, buoyancy_factor):
    if isinstance(gravity, (int, float)):
        gravity = np.array([gravity] + ([0] * (math.spatial_rank(density) - 1)))
    return StaggeredGrid.from_scalar(density, -gravity * buoyancy_factor)

