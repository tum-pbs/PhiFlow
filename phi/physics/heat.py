from phi import struct
from .physics import Physics, StateDependency
from .field.util import diffuse
from .field.effect import effect_applied
from .domain import DomainState


@struct.definition()
class Heat(DomainState):

    def __init__(self, domain, temperature=0.0, diffusivity=0.1, tags=('heat', 'pde', 'temperaturefield'), **kwargs):
        DomainState.__init__(self, **struct.kwargs(locals()))

    def default_physics(self):
        return HeatPhysics()

    @struct.variable(default=0.0, dependencies=DomainState.domain)
    def temperature(self, temperature):
        return self.centered_grid('temperature', temperature)

    @struct.constant(default=0.1)
    def diffusivity(self, diffusivity):
        return diffusivity


class HeatPhysics(Physics):

    def __init__(self):
        Physics.__init__(self, [StateDependency('effects', 'temperature_effect', blocking=True)])

    def step(self, heat, dt=1.0, effects=()):
        # pylint: disable-msg = arguments-differ
        temperature = diffuse(heat.temperature, dt * heat.diffusivity)
        for effect in effects:
            temperature = effect_applied(effect, temperature, dt)
        return heat.copied_with(temperature=temperature, age=heat.age+dt)
