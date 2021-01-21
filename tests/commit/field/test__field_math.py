from unittest import TestCase

from phi import field
from phi.physics import Domain


class TestFieldMath(TestCase):

    def test_gradient(self):
        domain = Domain(x=4, y=3)
        phi = domain.grid() * (1, 2)
        grad = field.gradient(phi)
        self.assertEqual(('spatial', 'spatial', 'channel', 'channel'), grad.shape.types)
