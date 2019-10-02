from .smoke import *
from .util import diffuse


class Burger(State):
    __struct__ = State.__struct__.extend(('_velocity',), ('_domain', '_viscosity',))

    def __init__(self, domain, velocity, viscosity=0.1, batch_size=None):
        State.__init__(self, tags=('burger', 'velocityfield'), batch_size=batch_size)
        self._domain = domain
        self._velocity = initialize_field(velocity, self.domain.shape(self.domain.rank, self._batch_size))
        self._viscosity = viscosity

    def default_physics(self):
        return BurgerPhysics()

    def copied_with(self, **kwargs):
        if 'velocity' in kwargs:
            kwargs['velocity'] = initialize_field(kwargs['velocity'], self.domain.shape(self.domain.rank, self._batch_size))
        return State.copied_with(self, **kwargs)

    @property
    def velocity(self):
        return self._velocity

    @property
    def domain(self):
        return self._domain

    @property
    def viscosity(self):
        return self._viscosity

    @property
    def centered_shape(self):
        return self.domain.shape(1, self._batch_size)


class BurgerPhysics(Physics):

    def __init__(self):
        Physics.__init__(self, {})

    def step(self, state, dt=1.0, **dependent_states):
        assert len(dependent_states) == 0
        v = advect(diffuse(state.velocity, dt * state.viscosity), dt)
        return state.copied_with(velocity=v, age=state.age + dt)


def advect(velocity, dt):
    idx = math.indices_tensor(velocity)
    sample_coords = idx - velocity * dt
    result = math.resample(velocity, sample_coords, interpolation='linear', boundary='REPLICATE')
    return result
