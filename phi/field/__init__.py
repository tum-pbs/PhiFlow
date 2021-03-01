"""
The fields module provides a number of data structures and functions to represent continuous, spatially varying data.

All fields are subclasses of Field which provides abstract functions for sampling field values at physical locations.

The most commonly used field types are

* CenteredGrid embeds a tensor in the physical space. Uses linear interpolation between grid points.
* StaggeredGrid samples the vector components at face centers instead of at cell centers.
* Noise is a function that produces a procedurally generated noise field

See the `phi.field` module documentation at https://tum-pbs.github.io/PhiFlow/Fields.html
"""

from ._field import Field, SampledField
from ._constant import ConstantField
from ._mask import HardGeometryMask, SoftGeometryMask as GeometryMask, SoftGeometryMask
from ._grid import Grid, CenteredGrid, StaggeredGrid, unstack_staggered_tensor, stack_staggered_components
from ._point_cloud import PointCloud
from ._noise import Noise
from ._angular_velocity import AngularVelocity
from ._field_math import (
    assert_close,
    laplace, spatial_gradient, divergence, stagger,  # spatial operators
    mean, pad, shift, normalize,
    concat, batch_stack,
    abs, sign, round_ as round, ceil, floor, sqrt, exp, isfinite, real, imag, sin, cos, cast, stop_gradient,  # op1
    solve, minimize,
    where,
    l2_loss,
    downsample2x, upsample2x,
    extrapolate_valid,
    jit_compile, functional_gradient,  # function wrappers
)
from ._field_io import write, read
from ._scene import Scene

__all__ = [key for key in globals().keys() if not key.startswith('_')]
