from unittest import TestCase

from phiml import math
from phi.geom import Sphere
from phi.physics._boundaries import Domain
from phi.field import *


class TestNoise(TestCase):

    def test_masks(self):
        domain = Domain(x=10, y=10)
        sphere = Sphere(x=5, y=5, radius=2)
        hard_v = domain.staggered_grid(HardGeometryMask(sphere))
        hard_s = domain.grid(HardGeometryMask(sphere))
        soft_v = domain.staggered_grid(SoftGeometryMask(sphere))
        soft_s = domain.grid(SoftGeometryMask(sphere))
        for f in [hard_v, hard_s, soft_v, soft_s]:
            math.assert_close(1, f.values.max)
            math.assert_close(0, f.values.min)
