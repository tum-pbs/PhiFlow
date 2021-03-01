from unittest import TestCase

import phi
from phi.math import NUMPY_BACKEND
from phi.math.extrapolation import *
from phi import math


BACKENDS = phi.detect_backends()


class TestExtrapolation(TestCase):

    def test_pad(self):
        test_in_func_out = [
            (math.zeros(x=3, y=4, z=5, a=1),
             lambda tensor: ConstantExtrapolation(0).pad(tensor, dict(x=[1, 1], y=[1, 0], z=[0, 1], a=[0, 0])),
             math.zeros(x=5, y=5, z=6, a=1)),
            (math.ones(x=3, y=4, z=5, a=1),
             lambda tensor: ConstantExtrapolation(1).pad(tensor, dict(x=[1, 1], y=[1, 0], z=[0, 1], a=[0, 0])),
             math.ones(x=5, y=5, z=6, a=1)),
            (-math.ones(x=3, y=4, z=5, a=1),
             lambda tensor: ConstantExtrapolation(-1).pad(tensor, dict(x=[1, 1], y=[1, 0], z=[0, 1], a=[0, 0])),
             - math.ones(x=5, y=5, z=6, a=1)),
        ]
        for val_in, func, val_out in test_in_func_out:
            try:
                math.assert_close(val_out, func(val_in))
                # TypeError('__bool__ should return bool, returned NotImplementedType')
                # self.assertEqual(val_out, func(val_in))
            except Exception as e:
                raise BaseException(AssertionError(e, val_in, func, val_out))


class TestExtrapolationOperators(TestCase):
    """ensures that proper propagation of extrapolation occurs (for Field arithmetics)"""

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
        try:
            PERIODIC + BOUNDARY
            assert False
        except TypeError:
            pass

        try:
            PERIODIC + ONE
            assert False
        except TypeError:
            pass

    def test_pad_tensor(self):
        for backend in BACKENDS:
            with backend:
                a = math.meshgrid(x=4, y=3)
                # 0
                p = math.pad(a, {'x': (1, 2), 'y': (0, 1)}, ZERO)
                self.assertEqual((7, 4, 2), p.shape.sizes)  # dimension check
                math.assert_close(p.x[1:-2].y[:-1], a)  # copy inner
                math.assert_close(p.x[0], 0)
                # 1
                p = math.pad(a, {'x': (1, 2), 'y': (0, 1)}, ONE)
                self.assertEqual((7, 4, 2), p.shape.sizes)  # dimension check
                math.assert_close(p.x[1:-2].y[:-1], a)  # copy inner
                math.assert_close(p.x[0], 1)
                # periodic
                p = math.pad(a, {'x': (1, 2), 'y': (0, 1)}, PERIODIC)
                self.assertEqual((7, 4, 2), p.shape.sizes)  # dimension check
                math.assert_close(p.x[1:-2].y[:-1], a)  # copy inner
                math.assert_close(p.x[0].y[:-1], a.x[-1])
                math.assert_close(p.x[-2:].y[:-1], a.x[:2])
                # boundary
                p = math.pad(a, {'x': (1, 2), 'y': (0, 1)}, BOUNDARY)
                self.assertEqual((7, 4, 2), p.shape.sizes)  # dimension check
                math.assert_close(p.x[1:-2].y[:-1], a)  # copy inner
                math.assert_close(p.x[0].y[:-1], a.x[0])
                math.assert_close(p.x[-2:].y[:-1], a.x[-1])
                # mixed
                p = math.pad(a, {'x': (1, 2), 'y': (0, 1)}, combine_sides({'x': PERIODIC, 'y': (ONE, REFLECT)}))
                math.print(p)
                self.assertEqual((7, 4, 2), p.shape.sizes)  # dimension check
                math.assert_close(p.x[1:-2].y[:-1], a)  # copy inner
                math.assert_close(p.x[0].y[:-1], a.x[-1])  # periodic
                math.assert_close(p.x[-2:].y[:-1], a.x[:2])  # periodic

    def test_pad_collapsed(self):
        a = math.zeros(b=2, x=10, y=10, batch=10)
        p = math.pad(a, {'x': (1, 2)}, ZERO)
        self.assertIsInstance(p, CollapsedTensor)
        self.assertEqual((10, 2, 13, 10), p.shape.sizes)
        p = math.pad(a, {'x': (1, 2)}, PERIODIC)
        self.assertIsInstance(p, CollapsedTensor)
        self.assertEqual((10, 2, 13, 10), p.shape.sizes)
