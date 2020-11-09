# pylint: disable-msg = unused-import
"""
Use this module as your main PhiFlow import.

from phi.flow import *

Contains:

* App class, show()
* phi.math as math
* Geometry objects
* phi.field and common Field classes
* Common physics functions such as diffuse, divergence_free, advect family
* I/O functions such as write_sim_frame
"""

import numpy
import numpy as np

from phi import math, struct
from phi.math import extrapolation, PI

from phi import geom
from phi.geom import Geometry, Sphere, Box, union

from phi import field
from phi.field import Grid, CenteredGrid, StaggeredGrid, GeometryMask, SoftGeometryMask, HardGeometryMask, Noise, PointCloud

from phi.physics import _boundaries as boundaries, fluid, _advect as advect
from phi.physics._boundaries import Domain, Material, OPEN, CLOSED, PERIODIC, NO_SLIP, NO_STICK, STICKY, SLIPPERY, Obstacle

from phi.data.fluidformat import write_sim_frame

from .app import App, EditableInt, EditableBool, EditableFloat, EditableString
from phi.viz import display
from phi.viz.display import show
