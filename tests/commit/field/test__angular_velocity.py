from unittest import TestCase

from phi.field import AngularVelocity, CenteredGrid, StaggeredGrid
from phi.math import channel, vec


class TestAngularVelocity(TestCase):

    def test_sample_at(self):
        field = AngularVelocity(location=vec(x=0, y=0))
        self.assertEqual(channel(vector='x,y'), field.shape.channel)
        field @ CenteredGrid(0, x=4, y=3)
        field @ StaggeredGrid(0, x=4, y=3)
