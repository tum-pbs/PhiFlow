from ._config import GLOBAL_AXIS_ORDER
from ._geom_util import assert_same_rank
from ._geom import Geometry
from ._union import union  # Union is private
from ._box import AABox, BoxGenerator as _BoxGenerator
from ._sphere import Sphere

box = _BoxGenerator()  # Instantiate an AABox using the syntax box[slices]
