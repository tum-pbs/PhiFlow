from unittest import TestCase

from phi.field import AngularVelocity, CenteredGrid, StaggeredGrid
from phiml.math import channel, vec


class TestAngularVelocity(TestCase):

    def test_sample_at(self):
        field = AngularVelocity(location=vec(x=0, y=0))
        field @ CenteredGrid(0, x=4, y=3)
        self.assertEqual(channel(vector='x,y'), (field @ CenteredGrid(0, x=4, y=3)).shape.channel)
        field @ StaggeredGrid(0, x=4, y=3)
