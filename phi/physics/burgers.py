import warnings

from phi import struct
from .field import advect
from .field.effect import effect_applied
from .field.util import diffuse
from .domain import DomainState
from .physics import Physics, StateDependency


@struct.definition()
class BurgersVelocity(DomainState):

    def __init__(self, domain, velocity=0, viscosity=0.1, name='burgers', **kwargs):
        DomainState.__init__(self, **struct.kwargs(locals()))

    @struct.variable(dependencies=DomainState.domain, default=0)
    def velocity(self, velocity):
        return self.centered_grid('velocity', velocity, self.rank)

    @struct.constant(default=0.1)
    def viscosity(self, viscosity):
        return viscosity

    def __add__(self, other):
        return self.copied_with(velocity=self.velocity + other)


class Burgers(Physics):

    def __init__(self, default_viscosity=0.1, viscosity=None, diffusion_substeps=1, advection=advect.semi_lagrangian):
        Physics.__init__(self, [StateDependency('effects', 'velocity_effect', blocking=True)])
        if viscosity is not None:
            warnings.warn("Argument 'viscosity' is deprecated, use 'default_viscosity' instead.", DeprecationWarning)
            default_viscosity = viscosity
        self.default_viscosity = default_viscosity
        self.diffusion_substeps = diffusion_substeps
        self.advection = advection

    def step(self, v, dt=1.0, effects=()):
        if isinstance(v, BurgersVelocity):
            return v.copied_with(velocity=self.step_velocity(v.velocity, v.viscosity, dt, effects, self.diffusion_substeps), age=v.age + dt)
        else:
            return self.step_velocity(v, self.default_viscosity, dt, effects, self.diffusion_substeps)

    def step_velocity(self, v, viscosity, dt, effects, diffusion_substeps):
        v = diffuse(v, dt * viscosity, substeps=diffusion_substeps)
        v = self.advection(v, v, dt)
        for effect in effects:
            v = effect_applied(effect, v, dt)
        return v.copied_with(age=v.age + dt)
