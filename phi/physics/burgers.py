from phi.field import advect
from .physics import *
from .util import diffuse
from .domain import *


class Burger(DomainState):
    __struct__ = DomainState.__struct__.extend(['_velocity'], ['_viscosity'])

    def __init__(self, domain, velocity, viscosity=0.1, batch_size=None):
        DomainState.__init__(self, domain, tags=('burger', 'velocityfield'), batch_size=batch_size)
        self._velocity = velocity
        self._viscosity = viscosity
        self.__validate__()

    def default_physics(self):
        return BurgerPhysics()

    def __validate_velocity__(self):
        self._velocity = self.centered_grid('velocity', self._velocity, components=self.rank)

    @property
    def velocity(self):
        return self._velocity

    @property
    def domain(self):
        return self._domain

    @property
    def viscosity(self):
        return self._viscosity


class BurgerPhysics(Physics):

    def __init__(self):
        Physics.__init__(self, {})

    def step(self, state, dt=1.0, **dependent_states):
        assert len(dependent_states) == 0
        v = state.velocity
        v = advect.semi_lagrangian(v, v, dt)
        v = diffuse(v, dt * state.viscosity)
        return state.copied_with(velocity=v, age=state.age + dt)

