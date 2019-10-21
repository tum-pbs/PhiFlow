from .smoke import *
from .util import diffuse


class Heat(State):

    __struct__ = State.__struct__.extend(('_temperature',), ('_domain', '_diffusivity'))

    def __init__(self, domain, temperature=0.0, diffusivity=0.1, batch_size=None):
        State.__init__(self, tags=('heat', 'pde', 'temperaturefield'), batch_size=batch_size)
        self._domain = domain
        self._temperature = domain.centered_grid(temperature, name='temperature', batch_size=self._batch_size)
        self._diffusivity = diffusivity

    def copied_with(self, **kwargs):
        if 'temperature' in kwargs:
            kwargs['temperature'] = self.domain.centered_grid(kwargs['temperature'], name='temperature', batch_size=self._batch_size)
        return State.copied_with(self, **kwargs)

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