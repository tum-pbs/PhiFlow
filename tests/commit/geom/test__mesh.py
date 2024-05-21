from unittest import TestCase

from phi import math
from phi.geom import Box, build_mesh, Sphere
from phiml.math import spatial


class TestGrid(TestCase):

    def test_build_mesh(self):
        build_mesh(Box(x=1, y=1), x=2, y=2)
        build_mesh(x=[0, 1, 3], y=[0, 1, 2])
        build_mesh(x=math.linspace(0, 1, spatial(x=10)), y=[0, 1, 4, 5])
        # --- with obstacles ---
        obs = Sphere(x=.5, y=0, radius=.5)
        build_mesh(Box(x=1, y=1), x=10, y=10, obstacles=obs)
