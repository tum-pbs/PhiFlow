from unittest import TestCase
from phi.field import *
from phi.math.geom import *


class TestMath(TestCase):

    def test_compatibility(self):
        f = CenteredGrid('f', box[0:3, 0:3], math.zeros([1,4,4,1]))
        g = CenteredGrid('g', box[0:3, 0:3], math.zeros([1,3,3,1]))
        self.assertTrue(f.points.compatible(f))
        self.assertFalse(f.compatible(g))
