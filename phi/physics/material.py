import math
from phi.math import Struct, struct


class Material(Struct):
    __struct__ = struct.Def((), ('solid', 'friction', 'extrapolate_fluid', 'global_velocity', 'local_velocity'))

    def __init__(self, solid=True, friction=0.0, extrapolate_fluid=True, global_velocity=0.0, local_velocity=0.0):
        """
Defines physical properties of a boundary or voxel.

Velocity:
The total velocity at a surface point is global_velocity + local_to_global(local_velocity).
If local_velocity is None, the latter term will be ignored.
        :param solid: Fluid can only enter non-solid cells or pass through non-solid boundaries
        :param friction: (only for solid materials) velocity decay rate in units of 1/time. 0: fluid can move parallell to the surface (no-stick), 1: fluid cannot move parallel (no-slip)
        :param extrapolate_fluid: Boundary condition when extrapolating the fluid channel into the object
        :param global_velocity: velocity offset in unmoving reference frame or 0 if unmoving.
        :param local_velocity: velocity offset in object reference frame. Set to 0 to add object's velocity, None to ignore completely.
        """
        self.solid = solid
        self.friction = friction
        self.extrapolate_fluid = extrapolate_fluid
        self.global_velocity = global_velocity
        self.local_velocity = local_velocity

    def friction_multiplier(self, dt=1):
        if dt == 1 or self.friction == 1 or self.friction == 0:
            return 1 - self.friction
        else:
            time_friction_exponent = math.log(1/(1-self.friction))
            return math.exp(- dt * time_friction_exponent)


OPEN = Material(solid=False, extrapolate_fluid=False, local_velocity=None)
NO_STICK = SLIPPERY = Material(solid=True, friction=0.0)
NO_SLIP = STICKY = Material(solid=True, friction=1.0)