from unittest import TestCase

from phi import math
from phi.field import AngularVelocity
from phi.physics import Domain


class TestAngularVelocity(TestCase):

    def test_sample_at(self):
        DOMAIN = Domain(x=4, y=3)
        field = AngularVelocity([0, 0])
        self.assertEqual(math.shape(vector=2), field.shape.channel)
        field >> DOMAIN.vector_grid()
        field >> DOMAIN.staggered_grid()
