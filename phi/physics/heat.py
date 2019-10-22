from .smoke import *
from .util import diffuse


class Heat(DomainState):
    __struct__ = State.__struct__.extend(('_temperature',), ('_domain', '_diffusivity'))

    def __init__(self, domain, temperature=0.0, diffusivity=0.1, batch_size=None):
        DomainState.__init__(self, domain, tags=('heat', 'pde', 'temperaturefield'), batch_size=batch_size)
        self._temperature = domain.centered_grid(temperature, name='temperature', batch_size=self._batch_size)
        self._diffusivity = diffusivity

    def __validate_temperature__(self):
        self._temperature = self.centered_grid('temperature', self._temperature)

    def default_physics(self):
        return HeatPhysics()

    @property
    def temperature(self):
        return self._temperature

    @property
    def domain(self):
        return self._domain

    @property
    def diffusivity(self):
        return self._diffusivity


class HeatPhysics(Physics):

    def __init__(self):
        Physics.__init__(self, blocking_dependencies={'effects': 'temperature_effect'})

    def step(self, heat, dt=1.0, effects=()):
        temperature = diffuse(heat.temperature, dt * heat.diffusivity)
        for effect in effects:
            temperature = effect_applied(effect, temperature, dt)
        return heat.copied_with(temperature=temperature, age=heat.age+dt)