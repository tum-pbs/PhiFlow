from .smoke import *


class Burger(State):
    __struct__ = State.__struct__.extend(('_velocity',), ('_domain', '_viscosity',))

    def __init__(self, domain, velocity, viscosity=0.1, batch_size=None):
        State.__init__(self, tags=('burger', 'velocityfield'), batch_size=batch_size)
        self._domain = domain
        self._velocity = initialize_field(velocity, self.grid.shape(self.grid.rank, self._batch_size))
        self._viscosity = viscosity

    def default_physics(self):
        return BurgerPhysics()

    def copied_with(self, **kwargs):
        if 'velocity' in kwargs:
            kwargs['velocity'] = initialize_field(kwargs['velocity'], self.grid.shape(self.grid.rank, self._batch_size))
        return State.copied_with(self, **kwargs)

    @property
    def velocity(self):
        return self._velocity

    @property
    def domain(self):
        return self._domain

    @property
    def grid(self):
        return self.domain.grid

    @property
    def viscosity(self):
        return self._viscosity

    @property
    def centered_shape(self):
        return self.grid.shape(1, self._batch_size)


class BurgerPhysics(Physics):

    def __init__(self):
        Physics.__init__(self, {})

    def step(self, state, dt=1.0, **dependent_states):
        assert len(dependent_states) == 0
        v = advect(diffuse(state.velocity, state.viscosity, dt), dt)
        return state.copied_with(velocity=v, age=state.age + dt)


def vector_laplace(v):
    return np.concatenate([laplace(v[...,i:i+1]) for i in range(v.shape[-1])], -1)


def advect(velocity, dt):
    idx = indices_tensor(velocity)
    sample_coords = idx - velocity * dt
    result = resample(velocity, sample_coords, interpolation='linear', boundary='REPLICATE')
    return result


def diffuse(velocity, viscosity, dt):
    return velocity + dt * viscosity * vector_laplace(velocity)