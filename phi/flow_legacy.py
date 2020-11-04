from .flow import *
from .physics._boundaries import Material, Obstacle, OPEN, CLOSED, NO_STICK, SLIPPERY, NO_SLIP, STICKY, PERIODIC, Domain, GeometryMovement
from .physics._effect import FieldEffect, GROW, ADD, FIX, Inflow, Accelerator, ConstantVelocity, HeatSource, ColdSource, Fan, Gravity
from .physics._world import World, world, obstacle_mask
from .physics.fluid import Fluid, IncompressibleFlow
from phi.field import diffuse

physics_config = math.GLOBAL_AXIS_ORDER
physics_config.x_last()
box = Box
SampledField = PointCloud
