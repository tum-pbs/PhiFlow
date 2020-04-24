from .field import Field, StaggeredSamplePoints, IncompatibleFieldTypes
from .flag import Flag, DIVERGENCE_FREE, L2_NORMALIZED
from .constant import ConstantField
from .grid import CenteredGrid
from .staggered_grid import StaggeredGrid, unstack_staggered_tensor
from .sampled import SampledField
from .analytic import AnalyticField, SymbolicFieldBackend
from .mask import GeometryMask, mask, union_mask
from .noise import Noise
from . import advect
from . import manta
from .util import diffuse, data_bounds, staggered_curl_2d


from phi import math

math.DYNAMIC_BACKEND.add_backend(SymbolicFieldBackend(math.DYNAMIC_BACKEND), priority=True)
