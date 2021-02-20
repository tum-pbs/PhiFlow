# pylint: disable-msg = unused-import
"""
*Compatibility import for PhiFlow 1 projects:* `from phi.flow_1 import *`
"""

from .flow import *
from .physics._boundaries import Material, Obstacle, OPEN, CLOSED, NO_STICK, SLIPPERY, NO_SLIP, STICKY, PERIODIC, Domain, GeometryMovement
from .physics._effect import FieldEffect, GROW, ADD, FIX, Inflow, Accelerator, ConstantVelocity, HeatSource, ColdSource, Fan, Gravity
from .physics._world import World, world, obstacle_mask
from .physics._fluid_legacy import Fluid, IncompressibleFlow
from .physics.diffuse import explicit as diffuse
from phi.field._scene import write_sim_frame

physics_config = math.GLOBAL_AXIS_ORDER
physics_config.x_last()
box = Box
SampledField = PointCloud
