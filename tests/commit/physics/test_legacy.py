from unittest import TestCase

from phi import math
from phi.geom import Sphere, Box
from phi.math import batch, channel
from phi.physics._boundaries import Domain, Obstacle


class TestLegacyPhysics(TestCase):

    def test_domain(self):
        dom = Domain(x=16, y=16)
        dom.scalar_grid(0)

