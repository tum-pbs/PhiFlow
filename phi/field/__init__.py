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
from ._analytic import AnalyticField
from ._constant import ConstantField
from ._mask import HardGeometryMask, SoftGeometryMask as GeometryMask, SoftGeometryMask
from ._grid import Grid, CenteredGrid, StaggeredGrid, unstack_staggered_tensor, stack_staggered_components, extp_cgrid, extp_sgrid
from ._point_cloud import PointCloud
from ._noise import Noise
from ._angular_velocity import AngularVelocity
from ._field_math import (
    laplace, gradient, divergence, stagger,
    mean, pad, shift, normalize,
    concat,
    real, imag,
    solve,
    where,
    l2_loss,
    stop_gradient,
)
from ._field_io import write, read

__all__ = [key for key in globals().keys() if not key.startswith('_')]
