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


def _is_div_free(velocity, is_div_free):
    assert is_div_free in (True, False, None)
    if isinstance(is_div_free, bool): return is_div_free
    if isinstance(velocity, Number): return True
    return False


class DenseFluid(State):
    __struct__ = State.__struct__.extend(['_velocity'], ['_domain', '_is_divergence_free'])

    def __init__(self, domain, velocity=0, is_divergence_free=None, batch_size=None):
        State.__init__(self, tags=['fluid', 'densefluid'], batch_size=batch_size)
        self._domain = domain
        self._velocity = initialize_field(velocity, self.domain.staggered_shape(self._batch_size))
        self._is_divergence_free = _is_div_free(velocity, is_divergence_free)

    def default_physics(self):
        return INCOMPRESSIBLE_FLOW

    def copied_with(self, **kwargs):
        if 'is_divergence_free' not in kwargs: kwargs['is_divergence_free'] = None
        if 'velocity' in kwargs:
            kwargs['is_divergence_free'] = _is_div_free(kwargs['velocity'], kwargs['is_divergence_free'])
            kwargs['velocity'] = initialize_field(kwargs['velocity'], self.domain.staggered_shape(self._batch_size))
        return State.copied_with(self, **kwargs)

    @property
    def domain(self):
        return self._domain

    @property
    def velocity(self):
        return self._velocity

    @property
    def is_divergence_free(self):
        return self._is_divergence_free



class FluidProperty(State):
    __struct__ = State.__struct__.extend(['_field'], ['_domain'])

    def __init__(self, domain, name, field=0.0, batch_size=None):
        State.__init__(self, tags=[name, 'fluidproperty'], batch_size=batch_size)
        self._name = name
        self._domain = domain
        self._field = initialize_field(field, domain.shape(1, self._batch_size))

    def default_physics(self):
        return PassiveFlow([self._name])

    def copied_with(self, **kwargs):
        if 'field' in kwargs:
            kwargs['field'] = initialize_field(kwargs['field'], self.domain.shape(1, self._batch_size))
        return State.copied_with(self, **kwargs)

    @property
    def domain(self):
        return self._domain

    @property
    def field(self):
        return self._field

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return 'fluid.%s' % self._name


def solve_pressure(obj, domaincache, pressure_solver=None):
    """
Calculates the pressure from the given velocity or velocity divergence using the specified solver.
    :param obj: tensor containing the centered velocity divergence values or velocity as StaggeredGrid
    :param solver: PressureSolver to use, options DEFAULT, SCIPY or MANTA
    :return: scalar pressure channel as tensor
    """
    if isinstance(obj, DenseFluid):
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


def divergence_free(velocity, domaincache, pressure_solver=None, return_pressure_and_iterations=False):
    assert isinstance(velocity, math.StaggeredGrid)
    velocity = domaincache.with_hard_boundary_conditions(velocity)
    pressure, iter = solve_pressure(velocity, domaincache, pressure_solver)
    gradp = math.StaggeredGrid.gradient(pressure)
    velocity -= domaincache.with_hard_boundary_conditions(gradp)
    if return_pressure_and_iterations:
        return velocity, (pressure, iter)
    else: return velocity


def _build_fluiddomain(domain, obstacles):
    mask = 1 - geometry_mask([o.geometry for o in obstacles], domain)
    return FluidDomain(domain, obstacles, active=mask, accessible=mask)


class FluidFlow(Physics):

    def __init__(self, incompressible=True, pressure_solver=None):
        Physics.__init__(self, dependencies={'obstacles': ['obstacle']},
                         blocking_dependencies={'velocity_effects': 'velocity_effect'})
        self.pressure_solver = pressure_solver
        self.incompressible = incompressible

    def step(self, densefluid, dt=1.0, obstacles=(), velocity_effects=()):
        dom = _build_fluiddomain(densefluid.domain, obstacles)
        velocity = densefluid.velocity
        if not densefluid.is_divergence_free and self.incompressible:
            velocity = divergence_free(velocity, dom, self.pressure_solver)
        # --- Advection ---
        velocity = velocity.advect(velocity, dt=dt)
        # --- velocity effects
        for effect in velocity_effects:
            velocity = effect.apply_grid(velocity, densefluid.domain, staggered=True, dt=dt)
        if self.incompressible:
            velocity = divergence_free(velocity, dom, self.pressure_solver)
        return densefluid.copied_with(velocity=velocity, is_divergence_free=True, age=densefluid.age + dt)


INCOMPRESSIBLE_FLOW = FluidFlow(True)
COMPRESSIBLE_FLOW = FluidFlow(False)


class PassiveFlow(Physics):

    def __init__(self, affected_by=()):
        Physics.__init__(self, {'fluids': 'fluid'},
                         blocking_dependencies={'effects': ['%s_effect'%str for str in affected_by]})

    def step(self, fluidproperty, dt=1.0, fluids=(), effects=()):
        field = fluidproperty.field
        for fluid in fluids:
            field = fluid.velocity.advect(field, dt=dt)
        # --- Effects ---
        for effect in effects:
            field = effect.apply_grid(field, fluidproperty.domain, staggered=False, dt=dt)
        return fluidproperty.copied_with(field=field, age=fluidproperty.age + dt)


class Buoyancy(FieldEffect):
    __struct__ = FieldEffect.__struct__.extend([], ['_strength'])

    def __init__(self, temperaturekey, strength=0.2, batch_size=None):
        FieldEffect.__init__(self, None, ['velocity'], batch_size=batch_size)
        self._strength = strength
        self._temperaturekey = temperaturekey if isinstance(temperaturekey, TrajectoryKey) else temperaturekey.trajectorykey

    @property
    def strength(self):
        return self._strength

    def default_physics(self):
        return BuoyancyPhysics()


class BuoyancyPhysics(Physics):

    def __init__(self):
        Physics.__init__(self, blocking_dependencies={'temperature_fields': 'temperature', 'gravity': 'gravity'})

    def step(self, buoyancy, dt=1.0, temperature_fields=(), gravity=()):
        gravity = sum(gravity, Gravity(0.0))
        for temperature in temperature_fields:
            if temperature.trajectorykey == buoyancy._temperaturekey:
                gravity = gravity.gravity_tensor(temperature.domain.rank)
                force_staggered = math.StaggeredGrid.from_scalar(temperature.field, gravity * buoyancy.strength)
                force = GridField(temperature.domain, force_staggered)
                return buoyancy.copied_with(field=force, age=buoyancy.age+dt)
        raise ValueError('Buoyancy source is not part of the simulation.')