import numpy as np

from phi import struct, math
from ._geom import Geometry


@struct.definition()
class _NoGeometry(Geometry):

    @property
    def center(self):
        return 0

    def bounding_radius(self):
        return 0

    def bounding_half_extent(self):
        return 0

    def rank(self):
        return None

    def approximate_signed_distance(self, location):
        return math.tile(np.inf, list(math.shape(location)[:-1]) + [1])

    def lies_inside(self, location):
        return math.tile(False, list(math.shape(location)[:-1]) + [1])

    def approximate_fraction_inside(self, location, cell_size):
        return math.tile(math.to_float(0), list(math.shape(location)[:-1]) + [1])

    def shifted(self, delta):
        return self

    def rotated(self, angle):
        return self


NO_GEOMETRY = _NoGeometry()
