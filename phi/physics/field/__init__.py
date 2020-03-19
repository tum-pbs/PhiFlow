from .field import Field, StaggeredSamplePoints, IncompatibleFieldTypes
from .flag import Flag, DIVERGENCE_FREE, L2_NORMALIZED
from .constant import ConstantField
from .grid import CenteredGrid
from .staggered_grid import StaggeredGrid, unstack_staggered_tensor
from .mask import GeometryMask, mask, union_mask
from .analytic import AnalyticField
from .sampled import SampledField
from .noise import Noise
from . import advect
from . import manta
from .util import diffuse, data_bounds
from .field_math import SymbolicFieldBackend


from phi import math

math.DYNAMIC_BACKEND.add_backend(SymbolicFieldBackend(), priority=True)
