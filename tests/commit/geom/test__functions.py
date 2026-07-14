from unittest import TestCase


from phi import math
from phi.field import CenteredGrid
from phi.geom import UniformGrid, Box
from phi.geom._functions import closest_on_triangle, closest_line_line
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

    def test_closest_line_line(self):
        # Skew lines with known closest point on line 1 at x=1.
        t = closest_line_line(
            offset1=vec(x=0, y=0, z=0),
            direction1=vec(x=1, y=0, z=0),
            offset2=vec(x=1, y=1, z=0),
            direction2=vec(x=0, y=1, z=1),
        )
        math.assert_close(1, t)

        # Same geometry with non-normalized directions changes only the parameter scale.
        t_scaled = closest_line_line(
            offset1=vec(x=0, y=0, z=0),
            direction1=vec(x=2, y=0, z=0),
            offset2=vec(x=1, y=1, z=0),
            direction2=vec(x=0, y=2, z=2),
        )
        math.assert_close(0.5, t_scaled)

        # Parallel lines return where_parallel.
        t_parallel = closest_line_line(
            offset1=vec(x=0, y=0, z=0),
            direction1=vec(x=1, y=0, z=0),
            offset2=vec(x=0, y=1, z=0),
            direction2=vec(x=2, y=0, z=0),
            where_parallel=0,
        )
        math.assert_close(0, t_parallel)

