from unittest import TestCase
from phi.math._extrapolation import *
from phi import math


class TestExtrapolationOperators(TestCase):

    def test_constant(self):
        self.assertEqual(ConstantExtrapolation(2), ONE + ONE)
        self.assertEqual(ZERO, ONE - ONE)
        self.assertEqual(ONE, ONE * ONE)
        self.assertEqual(ONE, ONE / ONE)
        self.assertEqual(ZERO, ZERO / ONE)

    def test_constant_periodic_working(self):
        self.assertEqual(PERIODIC, PERIODIC + ZERO)
        self.assertEqual(PERIODIC, PERIODIC - ZERO)
        self.assertEqual(PERIODIC, ZERO + PERIODIC)
        self.assertEqual(PERIODIC, PERIODIC / ONE)
        self.assertEqual(PERIODIC, PERIODIC * ONE)
        self.assertEqual(ZERO, PERIODIC * ZERO)

    def test_periodic_periodic(self):
        self.assertEqual(PERIODIC, PERIODIC + PERIODIC)
        self.assertEqual(PERIODIC, PERIODIC - PERIODIC)
        self.assertEqual(PERIODIC, PERIODIC * PERIODIC)
        self.assertEqual(PERIODIC, PERIODIC / PERIODIC)

    def test_cross_errors(self):
        try: PERIODIC + BOUNDARY; assert False
        except IncompatibleExtrapolations: pass

        try: PERIODIC + ONE; assert False
        except IncompatibleExtrapolations: pass

    def test_pad_tensor(self):
        a = math.meshgrid([1, 2, 3, 4], [5, 6, 7])
        extrap = MixedExtrapolation({'x': PERIODIC, 'y': (ONE, REFLECT)})
        p = math.pad(a, {'x': (1, 2), 'y': (3, 4)}, extrap)
        math.print(p)

    def test_pad_collapsed(self):
        a = math.zeros(2, x=10, y=10, batch=10)
        p = math.pad(a, {'x': (1, 2)}, ZERO)
        self.assertIsInstance(p, CollapsedTensor)
        self.assertEqual((10, 13, 10, 2), p.shape.sizes)
        p = math.pad(a, {'x': (1, 2)}, PERIODIC)
        self.assertIsInstance(p, CollapsedTensor)
        self.assertEqual((10, 13, 10, 2), p.shape.sizes)
