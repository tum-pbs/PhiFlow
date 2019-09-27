from .smoke import *


class Heat(State):

    __struct__ = State.__struct__.extend(('_temperature',), ('_domain', '_diffusivity'))

    def __init__(self, domain, temperature=0.0, diffusivity=0.1, batch_size=None):
        State.__init__(self, tags=('heat', 'pde', 'temperaturefield'), batch_size=batch_size)
        self._domain = domain
        self._temperature = temperature
        self._diffusivity = diffusivity

    def default_physics(self):
        return HeatPhysics()

    @property
    def temperature(self):
        return self._temperature

    @property
    def _temperature(self):
        return self._temperature_field

    @_temperature.setter
    def _temperature(self, value):
        self._temperature_field = initialize_field(value, self.domain.shape(1, self._batch_size))

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
        temperature = heat.temperature + heat.diffusivity * laplace(heat.temperature)
        for effect in effects:
            temperature = effect.apply_grid(temperature, heat.domain, staggered=False, dt=dt)
        return heat.copied_with(temperature=temperature, age=heat.age+dt)