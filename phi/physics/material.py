import math

from phi import struct


class Material(struct.Struct):

    def __init__(self, name, **kwargs):
        struct.Struct.__init__(**struct.kwargs(locals()))

    @struct.prop()
    def name(self, name):
        return str(name)

    @struct.prop(default=True)
    def solid(self, solid):
        """
Fluid can only enter non-solid cells or pass through non-solid boundaries.
        """
        assert isinstance(solid, bool)
        return solid

    @struct.prop(default=0.0)
    def friction(self, friction):
        """
(only for solid materials) velocity decay rate in units of 1/time.
0: fluid can move parallell to the surface (no-stick),
1: fluid cannot move parallel (no-slip)
        """
        return friction

    def friction_multiplier(self, dt=1):
        if dt == 1 or self.friction == 1 or self.friction == 0:
            return 1 - self.friction
        else:
            time_friction_exponent = math.log(1/(1-self.friction))
            return math.exp(- dt * time_friction_exponent)

    @struct.prop(default=False)
    def periodic(self, periodic):
        assert isinstance(periodic, bool)
        return periodic

    def __repr__(self):
        return self.name

    @property
    def extrapolation_mode(self):
        if self.periodic:
            return 'periodic'
        if self.solid:
            return 'boundary'
        else:
            return 'constant'


OPEN = Material('open', solid=False)
NO_STICK = SLIPPERY = Material('slippery', solid=True, friction=0.0)
NO_SLIP = STICKY = Material('sticky', solid=True, friction=1.0)
PERIODIC = Material('periodic', solid=False, periodic=True)