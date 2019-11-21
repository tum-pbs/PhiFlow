from unittest import TestCase

import numpy

from phi.geom import box
from phi.physics.domain import Domain, OPEN


class TestDomain(TestCase):

    def test_boundary_definitions(self):
        domain = Domain([128, 128, 16], (OPEN, OPEN, OPEN))
        self.assertEqual(domain.boundaries, OPEN)

        domain = Domain([64, 32], boundaries=[(OPEN, OPEN), (OPEN, OPEN)])
        self.assertEqual(domain.boundaries, OPEN)

        try:
            Domain([64, 32], None)
            self.fail()
        except AssertionError:
            pass

    def test_convenience_initializers(self):
        domain = Domain(64)
        numpy.testing.assert_equal(domain.resolution, [64])
        numpy.testing.assert_equal(domain.box.size, [64])

        domain = Domain(64, size=10)
        numpy.testing.assert_equal(domain.resolution, [64])
        numpy.testing.assert_equal(domain.box.size, [10])

        try:
            Domain(64, box=box[0:10], size=10)
            self.fail()
        except ValueError:
            pass