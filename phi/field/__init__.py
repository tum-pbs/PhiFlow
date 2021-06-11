"""
The fields module provides a number of data structures and functions to represent continuous, spatially varying data.

All fields are subclasses of `Field` which provides abstract functions for sampling field values at physical locations.

The most important field types are:

* `CenteredGrid` embeds a tensor in the physical space. Uses linear interpolation between grid points.
* `StaggeredGrid` samples the vector components at face centers instead of at cell centers.
* `Noise` is a function that produces a procedurally generated noise field

Use `grid()` to create a `Grid` from data or by sampling another `Field` or `phi.geom.Geometry`.
Alternatively, the `phi.physics.Domain` class provides convenience methods for grid creation.

All fields can be sampled at physical locations or volumes using `sample()` or `reduce_sample()`.

See the `phi.field` module documentation at https://tum-pbs.github.io/PhiFlow/Fields.html
"""

from ._field import Field, SampledField, unstack, sample, reduce_sample
from ._constant import ConstantField
from ._mask import HardGeometryMask, SoftGeometryMask as GeometryMask, SoftGeometryMask
from ._grid import Grid, CenteredGrid, StaggeredGrid
from ._point_cloud import PointCloud
from ._noise import Noise
from ._angular_velocity import AngularVelocity
from phi.math import (
    abs, sign, round, ceil, floor, sqrt, exp, isfinite, real, imag, sin, cos, cast, to_float, to_int32, to_int64, convert,
    stop_gradient,
    jit_compile, jit_compile_linear, functional_gradient,
    solve_linear, solve_nonlinear, minimize,
    l2_loss, l1_loss, frequency_loss,
)
from ._field_math import (
    assert_close,
    bake_extrapolation,
    laplace, spatial_gradient, divergence, stagger, curl,  # spatial operators
    fourier_poisson, fourier_laplace,
    mean, pad, shift, normalize, center_of_mass,
    concat, stack,
    where,
    vec_squared, vec_abs,
    downsample2x, upsample2x,
    extrapolate_valid,
    native_call,
    integrate,
)
from ._field_io import write, read
from ._scene import Scene

__all__ = [key for key in globals().keys() if not key.startswith('_')]

__pdoc__ = {
    'Grid.__init__': False,
    'Scene.__init__': False,
}
