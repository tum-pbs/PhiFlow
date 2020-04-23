"""
Surface material definitions including constants.
"""

import math

from phi import struct


@struct.definition()
class Material(struct.Struct):
    """
    Defines a surface material including the boundary conditions.
    """

    def __init__(self, name, **kwargs):
        struct.Struct.__init__(self, **struct.kwargs(locals()))

    @struct.constant()
    def name(self, name):
        """
Material name.
        """
        return str(name)

    @struct.constant(default=True)
    def solid(self, solid):
        """
Fluid can only enter non-solid cells or pass through non-solid boundaries.
        """
        assert isinstance(solid, bool)
        return solid

    @struct.derived()
    def open(self):
        return not self.solid and not self.periodic

    @struct.constant(default=0.0)
    def friction(self, friction):
        """
(only for solid materials) velocity decay rate in units of 1/time.
0: fluid can move parallell to the surface (no-stick),
1: fluid cannot move parallel (no-slip)
        """
        return friction

    def friction_multiplier(self, dt=1):
        """
Computes the velocity multiplication factor for fluid that moves along the surface for time dt.
        :param dt: time spent near surface (float)
        :return: factor (float)
        """
        if dt == 1 or self.friction == 1 or self.friction == 0:
            return 1 - self.friction
        else:
            time_friction_exponent = math.log(1 / (1 - self.friction))
            return math.exp(- dt * time_friction_exponent)

    @struct.constant(default=False)
    def periodic(self, periodic):
        """
Whether the boundary is periodic, i.e. seamlessly merges with the opposite end of the domain.
        """
        assert isinstance(periodic, bool)
        return periodic

    def __repr__(self):
        return self.name

    @struct.derived()
    def extrapolation_mode(self):
        """
Extrapolation mode for regular non-vector fields.
For vector fields that respect boundaries (e.g. velocity), use vector_extrapolation_mode.
    :return: one of ('periodic', 'boundary', 'constant').
    :rtype: str
        """
        if self.periodic:
            return 'periodic'
        if self.solid:
            return 'boundary'
        else:
            return 'constant'

    @struct.derived()
    def accessible_extrapolation_mode(self):
        if self.periodic:
            return 'periodic'
        if self.solid:
            return 'constant'
        else:
            return 'boundary'

    @struct.derived()
    def vector_extrapolation_mode(self):
        if self.periodic:
            return 'periodic'
        if self.solid:
            assert self.friction in (0, 1)
            return 'boundary' if self.friction == 0 else 'constant'
        else:
            return 'constant'


OPEN = Material('open', solid=False)
CLOSED = NO_STICK = SLIPPERY = Material('slippery', solid=True, friction=0)
NO_SLIP = STICKY = Material('sticky', solid=True, friction=1)
PERIODIC = Material('periodic', solid=False, periodic=True)
