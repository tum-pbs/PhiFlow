"""
The fields module provides a number of data structures and functions to represent continuous, spatially varying data.

All fields are subclasses of Field which provides abstract functions for sampling field values at physical locations.

The most commonly used field types are

* CenteredGrid embeds a tensor in the physical space. Uses linear interpolation between grid points.
* StaggeredGrid samples the vector components at face centers instead of at cell centers.
* Noise is a function that produces a procedurally generated noise field
"""

from ._field import Field, IncompatibleFieldTypes, SampledField
from ._analytic import AnalyticField, SymbolicFieldBackend
from ._constant import ConstantField
from ._mask import GeometryMask
from ._grid import Grid, CenteredGrid, StaggeredGrid, unstack_staggered_tensor, stack_staggered_components
from ._point_cloud import PointCloud
from ._noise import Noise
from ._angular_velocity import AngularVelocity
from ._field_math import (
    laplace, gradient, divergence, stagger, staggered_curl_2d,
    mean, pad, shift, normalize,
    expose_tensors,
    solve,
    divergence_free,
    diffuse,
)
from ._field_io import write, read

from phi import math as _math
_math.DYNAMIC_BACKEND.add_backend(SymbolicFieldBackend(_math.DYNAMIC_BACKEND), priority=True)
