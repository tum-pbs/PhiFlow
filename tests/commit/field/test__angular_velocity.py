from unittest import TestCase

from phi.field import AngularVelocity, CenteredGrid, StaggeredGrid
from phi.math import channel


class TestAngularVelocity(TestCase):

    def test_sample_at(self):
        field = AngularVelocity([0, 0])
        self.assertEqual(channel(vector=2), field.shape.channel)
        field @ CenteredGrid(0, x=4, y=3)
        field @ StaggeredGrid(0, x=4, y=3)
