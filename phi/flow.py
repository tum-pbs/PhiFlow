# pylint: disable-msg = wildcard-import, unused-wildcard-import, unused-import

from .physics.physics import *
from .physics.world import *
from .physics.schroedinger import *
from .physics.fluid import *
from .physics.burgers import *
from .physics.heat import *
from .physics.worldutil import *
from .physics.field import *
from .physics.obstacle import *
from .physics.material import *
from .physics.domain import *
from .physics.field.effect import *
from .physics.pressuresolver.solver_api import PoissonDomain, PoissonSolver
from .physics.pressuresolver.sparse import SparseCG, SparseSciPy
from .physics.pressuresolver.geom import GeometricCG
from .physics.pressuresolver.fourier import FourierSolver

from .data.fluidformat import *
from .data.dataset import *
from .data.stream import *
from .data.reader import *

from phi.geom import *
from phi import math, struct

from .viz import display
from .viz.display import show
from .app import *

import numpy

np = numpy

physics_config = GLOBAL_AXIS_ORDER
