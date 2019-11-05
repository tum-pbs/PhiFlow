from .smoke import *
from phi.physics.field.util import diffuse


class Heat(DomainState):

    def __init__(self, domain, temperature=0.0, diffusivity=0.1, tags=('heat', 'pde', 'temperaturefield'), **kwargs):
        DomainState.__init__(**struct.kwargs(locals()))

    def default_physics(self):
        return HeatPhysics()

    @struct.attr(default=0.0)
    def temperature(self, t):
        return self.centered_grid('temperature', t)

    @struct.prop(default=0.1)
    def diffusivity(self, d): return d


class HeatPhysics(Physics):

    def __init__(self):
        Physics.__init__(self, [StateDependency('effects', 'temperature_effect', blocking=True)])

    def step(self, heat, dt=1.0, effects=()):
        temperature = diffuse(heat.temperature, dt * heat.diffusivity)
        for effect in effects:
            temperature = effect_applied(effect, temperature, dt)
        return heat.copied_with(temperature=temperature, age=heat.age+dt)
