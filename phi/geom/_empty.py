import numpy as np

from phi import struct, math
from ._geom import Geometry


@struct.definition()
class _NoGeometry(Geometry):

    def rank(self):
        return None

    def approximate_signed_distance(self, location):
        return math.tile(np.inf, list(math.shape(location)[:-1]) + [1])

    def lies_inside(self, location):
        return math.tile(False, list(math.shape(location)[:-1]) + [1])

    def approximate_fraction_inside(self, location, cell_size):
        return math.tile(math.to_float(0), list(math.shape(location)[:-1]) + [1])


NO_GEOMETRY = _NoGeometry()
