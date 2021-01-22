from unittest import TestCase

from phi import math
from phi.geom import Box
from phi import field
from phi.physics import Domain


class TestFieldMath(TestCase):

    def test_gradient(self):
        domain = Domain(x=4, y=3)
        phi = domain.grid() * (1, 2)
        grad = field.gradient(phi, stack_dim='gradient')
        self.assertEqual(('spatial', 'spatial', 'channel', 'channel'), grad.shape.types)

    def test_divergence_centered(self):
        v = field.CenteredGrid(math.ones(x=3, y=3), Box[0:1, 0:1], math.extrapolation.ZERO) * (1, 0)  # flow to the right
        div = field.divergence(v).values
        math.assert_close(div.y[0], (1.5, 0, -1.5))
