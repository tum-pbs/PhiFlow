from unittest import TestCase

from phi import math
from phi.field import AngularVelocity
from phi.math import channel
from phi.physics._boundaries import Domain


class TestAngularVelocity(TestCase):

    def test_sample_at(self):
        DOMAIN = Domain(x=4, y=3)
        field = AngularVelocity([0, 0])
        self.assertEqual(channel(vector=2), field.shape.channel)
        field @ DOMAIN.vector_grid()
        field @ DOMAIN.staggered_grid()
