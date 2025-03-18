# pylint: disable-msg = unused-import
"""
*Main PhiFlow import:* `from phi.flow import *`

Imports important functions and classes from
`math`, `geom`, `field`, `physics` and `vis` (including sub-modules)
as well as the modules and sub-modules themselves.

See `phi.tf.flow`, `phi.torch.flow`, `phi.jax.flow`.
"""

# Modules
import numpy
import numpy as np
import phiml
from phiml import *
import phi
from . import geom, field, physics, vis
from .physics import fluid, advect, diffuse

# Classes
from .geom import Geometry, Point, Sphere, Box, Cuboid, cylinder, UniformGrid, Mesh, Graph
from .field import Field, Grid, CenteredGrid, StaggeredGrid, mask, Noise, PointCloud, Scene, resample, GeometryMask, SoftGeometryMask, HardGeometryMask
from .physics.fluid import Obstacle

# Functions
from .geom import union, rotate, scale, length, squared_length, normalize, cross
from .vis import show, control, plot
