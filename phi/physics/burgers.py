from .physics import Physics, StateDependency
from .field import advect
from .field.util import diffuse
from .field.effect import effect_applied


class Burgers(Physics):

    def __init__(self, viscosity=0.1):
        Physics.__init__(self, [StateDependency('effects', 'velocity_effect', blocking=True)])
        self.viscosity = viscosity

    def step(self, v, dt=1.0, effects=()):
        v = advect.semi_lagrangian(v, v, dt)
        v = diffuse(v, dt * self.viscosity, substeps=1)
        for effect in effects:
            v = effect_applied(effect, v, dt)
        return v.copied_with(age=v.age + dt)
