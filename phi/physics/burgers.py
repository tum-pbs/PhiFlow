from .field import advect
from .domain import DomainState
from .physics import Physics, StateDependency
from .field.util import diffuse
from .field.effect import effect_applied
from phi import struct


@struct.definition()
class Burgers(DomainState):

    def __init__(self, domain, velocity=0, viscosity=0.1, tags=('burgers', 'velocityfield'),
                 name='burgers', **kwargs):
        DomainState.__init__(self, **struct.kwargs(locals()))

    def default_physics(self):
        return BurgersPhysics()

    @struct.variable(default=0.0, dependencies=DomainState.domain)
    def velocity(self, velocity):
        return self.centered_grid('velocity', velocity, components=self.rank)

    @struct.constant(default=0.1)
    def viscosity(self, viscosity): return viscosity


class BurgersPhysics(Physics):

    def __init__(self):
        Physics.__init__(self, [StateDependency('effects', 'velocity_effect', blocking=True)])

    def step(self, state, dt=1.0, effects=()):
        v = state.velocity
        v = advect.semi_lagrangian(v, v, dt)
        v = diffuse(v, dt * state.viscosity, substeps=1)
        for effect in effects:
            v = effect_applied(effect, v, dt)
        return state.copied_with(velocity=v, age=state.age + dt)
