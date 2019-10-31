from phi.physics.field import advect

from .domain import DomainState
from .physics import Physics
from .field.util import diffuse


class Burgers(DomainState):
    __struct__ = DomainState.__struct__.extend(['_velocity'], ['_viscosity'])

    def __init__(self, domain, velocity, viscosity=0.1, batch_size=None):
        DomainState.__init__(self, domain, tags=('burgers', 'velocityfield'), batch_size=batch_size)
        self._velocity = velocity
        self._viscosity = viscosity
        self.__validate__()

    def default_physics(self):
        return BurgersPhysics()

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


class BurgersPhysics(Physics):

    def __init__(self):
        Physics.__init__(self)

    def step(self, state, dt=1.0, **dependent_states):
        assert len(dependent_states) == 0
        v = state.velocity
        v = advect.semi_lagrangian(v, v, dt)
        v = diffuse(v, dt * state.viscosity)
        return state.copied_with(velocity=v, age=state.age + dt)
