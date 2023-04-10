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
from .physics import fluid, advect, diffuse

# Classes
from .math import Shape, Tensor, DType, Solve
from .geom import Geometry, Sphere, Box, Cuboid
from .field import Field, Grid, CenteredGrid, StaggeredGrid, mask, Noise, PointCloud, Scene, resample, GeometryMask, SoftGeometryMask, HardGeometryMask
from .vis import Viewer
from .physics.fluid import Obstacle

# Constants
from .math import PI, INF, NAN, f
from .math.extrapolation import PERIODIC, ZERO_GRADIENT

# Functions
from .math import (
    wrap, tensor, vec,  # Tensor creation
    shape, spatial, channel, batch, instance, dual,
    non_spatial, non_channel, non_batch, non_instance, non_dual,  # Shape functions (magic)
    unstack, stack, concat, expand, rename_dims, pack_dims, unpack_dim, flatten, cast,  # Magic Ops
    jit_compile, jit_compile_linear, minimize, functional_gradient, solve_linear, solve_nonlinear, iterate, identity,  # jacobian, hessian, custom_gradient # Functional magic
)
from .geom import union
from .vis import show, view, control, plot

# Exceptions
from .math import ConvergenceException, NotConverged, Diverged
