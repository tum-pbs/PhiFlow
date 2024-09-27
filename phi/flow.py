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
from . import math
from phiml import backend
from phiml.math import extrapolation
import phi
from . import geom, field, physics, vis
from .physics import fluid, advect, diffuse

# Classes
from phiml.math import Shape, Tensor, DType, Solve
from .geom import Geometry, Point, Sphere, Box, Cuboid, UniformGrid, Mesh, Graph
from .field import Field, Grid, CenteredGrid, StaggeredGrid, mask, Noise, PointCloud, Scene, resample, GeometryMask, SoftGeometryMask, HardGeometryMask
from .vis import Viewer
from .physics.fluid import Obstacle

# Constants
from phiml.math import PI, INF, NAN, f
from phiml.math.extrapolation import PERIODIC, ZERO_GRADIENT

# Functions
from phiml.math import (
    wrap, tensor, vec, zeros, zeros_like, ones, ones_like, linspace,  # Tensor creation
    shape, spatial, channel, batch, instance, dual, primal,
    non_spatial, non_channel, non_batch, non_instance, non_dual, non_primal,  # Shape functions (magic)
    unstack, stack, concat, expand, rename_dims, pack_dims, unpack_dim, flatten, cast,  # Magic Ops
    b2i, c2b, c2d, i2b, s2b, si2d, d2i, d2s, map_s2b, map_i2b, map_c2b, map_d2c,  # dim type conversions
    sign, round, ceil, floor, sqrt, exp, erf, log, log2, log10, sigmoid, soft_plus,
    sin, cos, tan, sinh, cosh, tanh, arcsin, arccos, arctan, arcsinh, arccosh, arctanh, log_gamma, factorial, incomplete_gamma,
    scatter, gather,
    rotate_vector as rotate, cross_product as cross, dot, convolve, vec_normalize as normalize, length, maximum, minimum, clip,  # vector math
    safe_div, length, is_finite, is_nan, is_inf,  # Basic functions
    jit_compile, jit_compile_linear, minimize, gradient as functional_gradient, gradient, solve_linear, solve_nonlinear, iterate, identity,  # jacobian, hessian, custom_gradient # Functional magic
    assert_close, always_close, equal, close
)
from .geom import union
from .vis import show, view, control, plot

# Exceptions
from phiml.math import ConvergenceException, NotConverged, Diverged
