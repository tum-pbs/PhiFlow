from .field.effect import effect_applied
from .field.util import diffuse
from .physics import Physics, StateDependency


class HeatDiffusion(Physics):

    def __init__(self, diffusivity=0.1):
        Physics.__init__(self, [StateDependency('effects', 'temperature_effect', blocking=True)])
        self.diffusivity = diffusivity

    def step(self, temperature, dt=1.0, effects=()):
        # pylint: disable-msg = arguments-differ
        temperature = diffuse(temperature, dt * self.diffusivity)
        for effect in effects:
            temperature = effect_applied(effect, temperature, dt)
        return temperature.copied_with(age=temperature.age + dt)
