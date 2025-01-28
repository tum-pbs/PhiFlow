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
from .geom import Geometry, Point, Sphere, Box, Cuboid, cylinder, UniformGrid, Mesh, Graph
from .field import Field, Grid, CenteredGrid, StaggeredGrid, mask, Noise, PointCloud, Scene, resample, GeometryMask, SoftGeometryMask, HardGeometryMask
from .physics.fluid import Obstacle

# Constants
from phiml.math import PI, INF, NAN, f
from phiml.math.extrapolation import PERIODIC, ZERO_GRADIENT

# Functions
from phiml.math import (
    wrap, tensor, vec, zeros, zeros_like, ones, ones_like, linspace, rand, randn, arange, meshgrid,  # Tensor creation
    shape, spatial, channel, batch, instance, dual, primal,
    non_spatial, non_channel, non_batch, non_instance, non_dual, non_primal,  # Shape functions (magic)
    unstack, stack, concat, tcat, dcat, icat, scat, ccat, expand, rename_dims, pack_dims, dpack, ipack, spack, cpack, unpack_dim, flatten, cast,  # Magic Ops
    b2i, c2b, c2d, i2b, s2b, si2d, p2d, d2i, d2s, map_s2b, map_i2b, map_c2b, map_d2b, map_d2c, map_c2d,  # dim type conversions
    dsum, psum, isum, ssum, csum, mean, dmean, pmean, imean, smean, cmean, median, sign, round, ceil, floor, sqrt, exp, erf, log, log2, log10, sigmoid, soft_plus,
    dprod, pprod, sprod, iprod, cprod, dmin, pmin, smin, imin, cmin, finite_min, dmax, pmax, smax, imax, cmax, finite_max,

    sin, cos, tan, sinh, cosh, tanh, arcsin, arccos, arctan, arcsinh, arccosh, arctanh, log_gamma, factorial, incomplete_gamma,
    scatter, gather, where, nonzero,
    dot, convolve, maximum, minimum, clip,  # vector math
    safe_div, is_finite, is_nan, is_inf,  # Basic functions
    jit_compile, jit_compile_linear, minimize, gradient as functional_gradient, gradient, solve_linear, solve_nonlinear, iterate, identity,  # jacobian, hessian, custom_gradient # Functional magic
    assert_close, always_close, equal, close,
    l1_loss, l2_loss,
)
from .geom import union, rotate, scale, length, squared_length, normalize, cross
from .vis import show, control, plot

# Exceptions
from phiml.math import ConvergenceException, NotConverged, Diverged
