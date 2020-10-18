import numpy as np

from phi import math
from ..math._shape import EMPTY_SHAPE
from ._geom import Geometry


class _NoGeometry(Geometry):

    @property
    def shape(self):
        return EMPTY_SHAPE

    @property
    def center(self):
        return 0

    def bounding_radius(self):
        return 0

    def bounding_half_extent(self):
        return 0

    def approximate_signed_distance(self, location):
        return math.zeros(location.shape.non_channel) + np.inf

    def lies_inside(self, location):
        return math.zeros(location.shape.non_channel, dtype=bool)

    def approximate_fraction_inside(self, other_geometry):
        return math.zeros(other_geometry.shape)

    def shifted(self, delta):
        return self

    def rotated(self, angle):
        return self


NO_GEOMETRY = _NoGeometry()
