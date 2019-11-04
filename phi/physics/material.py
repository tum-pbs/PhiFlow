import math

from phi import struct


class Material(struct.Struct):

    def __init__(self, name, solid=True, friction=0.0, **kwargs):
        """
        Defines physical properties of a boundary or voxel.

        Velocity:
        The total velocity at a surface point is global_velocity + local_to_global(local_velocity).
        If local_velocity is None, the latter term will be ignored.
        :param solid: Fluid can only enter non-solid cells or pass through non-solid boundaries
        :param friction: (only for solid materials) velocity decay rate in units of 1/time. 0: fluid can move parallell to the surface (no-stick), 1: fluid cannot move parallel (no-slip)
        """
        struct.Struct.__init__(**struct.kwargs(locals()))

    @struct.prop()
    def name(self, name): return name

    @struct.prop(default=True)
    def solid(self, solid): return solid

    @struct.prop(default=0.0)
    def friction(self, friction): return friction

    def friction_multiplier(self, dt=1):
        if dt == 1 or self.friction == 1 or self.friction == 0:
            return 1 - self.friction
        else:
            time_friction_exponent = math.log(1/(1-self.friction))
            return math.exp(- dt * time_friction_exponent)

    def __repr__(self):
        return self.name


OPEN = Material('open', solid=False)
NO_STICK = SLIPPERY = Material('slippery', solid=True, friction=0.0)
NO_SLIP = STICKY = Material('sticky', solid=True, friction=1.0)
