from unittest import TestCase


from phi import math
from phi.field import CenteredGrid
from phi.geom import UniformGrid, Box
from phi.geom._functions import closest_on_triangle
from phi.math import batch, channel
from phi.math.magic import Shaped, Sliceable, Shapable
from phiml.math import vec, spatial


class TestGrid(TestCase):

    def test_closest_on_triangle(self):
        triangle = [vec(x=0, y=0, z=0), vec(x=1, y=0, z=0), vec(x=0, y=1, z=0)]
        def offset(x):
            return closest_on_triangle(*triangle, query=x, exact_edges=False) - x
        v = CenteredGrid(offset, x=32, y=32, z=1, bounds=Box(x=(-1.2, 2.2), y=(-1.2, 2.2), z=(0, 1)))
        # show(v.z[0].as_points())
