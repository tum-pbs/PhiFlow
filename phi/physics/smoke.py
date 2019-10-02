from .domain import *
from phi.solver.base import *
from phi.math.initializers import _is_python_shape, zeros


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


def domain(smoke, obstacles):
    if smoke.domaincache is None or not smoke.domaincache.is_valid(obstacles):
        mask = 1 - geometry_mask([o.geometry for o in obstacles], smoke.domain)
        smoke.domaincache = FluidDomain(smoke.domain, obstacles, active=mask, accessible=mask)
    return smoke.domaincache


class SmokePhysics(Physics):

    def __init__(self, pressure_solver=None, make_input_divfree=False, make_output_divfree=True):
        Physics.__init__(self, dependencies={'obstacles': ['obstacle']},
                         blocking_dependencies={'density_effects': 'density_effect', 'velocity_effects': 'velocity_effect'})
        self.pressure_solver = pressure_solver
        self.make_input_divfree = make_input_divfree
        self.make_output_divfree = make_output_divfree

    def step(self, smoke, dt=1.0, obstacles=(), density_effects=(), velocity_effects=(), **dependent_states):
        assert len(dependent_states) == 0
        domaincache = domain(smoke, obstacles)
        velocity = smoke.velocity
        density = smoke.density
        if self.make_input_divfree:
            velocity = divergence_free(velocity, domaincache, self.pressure_solver, smoke=smoke)
        # --- Advection ---
        density = velocity.advect(density, dt=dt)
        velocity = velocity.advect(velocity, dt=dt)
        # --- Density effects ---
        for effect in density_effects:
            density = effect.apply_grid(density, smoke.domain, staggered=False, dt=dt)
        # --- velocity effects
        for effect in velocity_effects:
            velocity = effect.apply_grid(velocity, smoke.domain, staggered=True, dt=dt)
        velocity = stick(velocity, domaincache, dt)
        velocity += dt * buoyancy(smoke.density, smoke.gravity, smoke.buoyancy_factor)
        if self.make_output_divfree:
            velocity = divergence_free(velocity, domaincache, self.pressure_solver, smoke=smoke)
        return smoke.copied_with(density=density, velocity=velocity, age=smoke.age + dt)


SMOKE = SmokePhysics()



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

    def __add__(self, other):
        if isinstance(other, math.StaggeredGrid):
            return self.copied_with(velocity=self.velocity + other)
        else:
            return self.copied_with(density=self.density + other)

    def __sub__(self, other):
        if isinstance(other, math.StaggeredGrid):
            return self.copied_with(velocity=self.velocity - other)
        else:
            return self.copied_with(density=self.density - other)



def solve_pressure(obj, domaincache, pressure_solver=None):
    """
Calculates the pressure from the given velocity or velocity divergence using the specified solver.
    :param obj: tensor containing the centered velocity divergence values or velocity as StaggeredGrid
    :param solver: PressureSolver to use, options DEFAULT, SCIPY or MANTA
    :return: scalar pressure channel as tensor
    """
    if isinstance(obj, Smoke):
        div = obj.velocity.divergence()
    elif isinstance(obj, math.StaggeredGrid):
        div = obj.divergence()
    elif obj.shape[-1] == domaincache.rank:
        div = math.divergence(obj, difference='central')
    else:
        raise ValueError("Cannot solve pressure for %s" % obj)

    if pressure_solver is None:
        from phi.solver.sparse import SparseCG
        pressure_solver = SparseCG()

    pressure, iter = pressure_solver.solve(div, domaincache, pressure_guess=None)
    return pressure, iter


def divergence_free(obj, domaincache, pressure_solver=None, smoke=None):
    if isinstance(obj, Smoke):
        return obj.copied_with(velocity=divergence_free(obj.velocity, domaincache))
    assert isinstance(obj, math.StaggeredGrid)
    velocity = obj
    velocity = domaincache.with_hard_boundary_conditions(velocity)
    pressure, iter = solve_pressure(velocity, domaincache, pressure_solver)
    gradp = math.StaggeredGrid.gradient(pressure)
    velocity -= domaincache.with_hard_boundary_conditions(gradp)
    if smoke is not None:
        smoke._last_pressure = pressure
        smoke._last_pressure_iterations = iter
    return velocity


def buoyancy(density, gravity, buoyancy_factor):
    if isinstance(gravity, (int, float)):
        gravity = np.array([gravity] + ([0] * (math.spatial_rank(density) - 1)))
    return math.StaggeredGrid.from_scalar(density, -gravity * buoyancy_factor)


def stick(velocity, domaincache, dt):
    velocity = domaincache.with_hard_boundary_conditions(velocity)
    # TODO wall friction
    # self.world.geom
    # friction = material.friction_multiplier(dt)
    return velocity


    # def serialize_to_dict(self):
    #     return {
    #         'type': 'Smoke',
    #         'class': self.__class__.__name__,
    #         'module': self.__class__.__module__,
    #         'rank': self.domain.rank,
    #         'domain': self.domain.serialize_to_dict(),
    #         'gravity': list(self.gravity),
    #         'buoyancy_factor': self.buoyancy_factor,
    #         'conserve_density': self.conserve_density,
    #         'solver': self.pressure_solver.name,
    #     }
