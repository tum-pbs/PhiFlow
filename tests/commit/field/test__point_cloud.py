from unittest import TestCase

from phi.field import PointCloud
from phi.geom import Sphere
from phi.math import batch, stack, instance, expand, rename_dims, shape, vec


class GridTest(TestCase):

    def test_reshape(self):
        c = PointCloud(Sphere(stack([vec(x=0, y=1)] * 50, instance('points')), radius=.1))
        c = expand(c, batch(b=2))
        c = rename_dims(c, 'points', 'particles')
        assert batch(b=2) & instance(particles=50) == shape(c)
