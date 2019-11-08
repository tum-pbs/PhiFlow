from .field import advect
from .domain import DomainState
from .physics import Physics
from .field.util import diffuse
from phi import struct


class Burgers(DomainState):

    def __init__(self, domain, velocity, viscosity=0.1, periodic=True, tags=('burgers', 'velocityfield'), **kwargs):
        DomainState.__init__(**struct.kwargs(locals()))

    def default_physics(self):
        return BurgersPhysics()

    @struct.attr(default=0.0)
    def velocity(self, velocity):
        velocity = self.centered_grid('velocity', velocity, components=self.rank)
        if self.periodic:
            velocity = velocity.copied_with(extrapolation='periodic')
        return velocity

    @struct.prop(default=0.1)
    def viscosity(self, viscosity): return viscosity

    @struct.prop(default=True)
    def periodic(self, periodic):
        assert isinstance(periodic, bool)
        return periodic


class BurgersPhysics(Physics):

    def __init__(self):
        Physics.__init__(self)

    def step(self, state, dt=1.0, **dependent_states):
        assert len(dependent_states) == 0
        v = state.velocity
        v = advect.semi_lagrangian(v, v, dt)
        v = diffuse(v, dt * state.viscosity, substeps=1)
        return state.copied_with(velocity=v, age=state.age + dt)
