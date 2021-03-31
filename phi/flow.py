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
import phi
from . import math, geom, field, physics, vis
from .math import extrapolation, backend
from .physics import fluid, flip, advect, diffuse

# Classes
from .math import DType
from .geom import Geometry, Sphere, Box
from .field import Grid, CenteredGrid, StaggeredGrid, GeometryMask, SoftGeometryMask, HardGeometryMask, Noise, PointCloud, Scene
from .physics import Domain, Obstacle
from .vis import view, Viewer, control

# Constants
from .math import PI
from .physics import OPEN, CLOSED, PERIODIC

# Functions
from .math import wrap, tensor, shape
from .geom import union
from .vis import show

# Exceptions
from .math import ConvergenceException, NotConverged, Diverged
