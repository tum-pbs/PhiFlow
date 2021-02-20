# pylint: disable-msg = unused-import
"""
*Main PhiFlow import:* `from phi.flow import *`

Imports important functions and classes from
`math`, `geom`, `field`, `physics` and `app` (including sub-modules)
as well as the modules and sub-modules themselves.

See `phi.tf.flow`, `phi.torch.flow`.
"""

import numpy
import numpy as np

from . import math
from .math import extrapolation, PI, DType, tensor, shape, backend

from . import geom
from .geom import Geometry, Sphere, Box, union

from . import field
from .field import Grid, CenteredGrid, StaggeredGrid, GeometryMask, SoftGeometryMask, HardGeometryMask, Noise, PointCloud, Scene

from . import physics
from .physics import fluid, flip, advect, diffuse, Domain, Material, OPEN, CLOSED, PERIODIC, Obstacle

from . import app
from .app import App, EditableInt, EditableBool, EditableFloat, EditableString, ModuleViewer, show
