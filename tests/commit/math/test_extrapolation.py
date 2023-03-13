from unittest import TestCase

import phi
from phi import math
from phi.math import batch, extrapolation, shape, spatial, channel, EMPTY_SHAPE
from phi.math._tensors import wrap
from phi.math.extrapolation import ConstantExtrapolation, ONE, ZERO, PERIODIC, BOUNDARY, SYMMETRIC, REFLECT, combine_sides, from_dict, combine_by_direction, SYMMETRIC_GRADIENT, as_extrapolation, \
    ZERO_GRADIENT

BACKENDS = phi.detect_backends()


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
            _ = PERIODIC + BOUNDARY
            self.fail("periodic and boundary are not compatible, should raise a TypeError")
        except TypeError:
            pass

        try:
            _ = PERIODIC + ONE
            self.fail("periodic and constant are not compatible, should raise a TypeError")
        except TypeError:
            pass


class TestExtrapolation(TestCase):

    def test_pad(self):
        test_in_func_out = [
            (math.zeros(spatial(x=3, y=4, z=5, a=1)),
             lambda tensor: ConstantExtrapolation(0).pad(tensor, dict(x=[1, 1], y=[1, 0], z=[0, 1], a=[0, 0])),
             math.zeros(spatial(x=5, y=5, z=6, a=1))),
            (math.ones(spatial(x=3, y=4, z=5, a=1)),
             lambda tensor: ConstantExtrapolation(1).pad(tensor, dict(x=[1, 1], y=[1, 0], z=[0, 1], a=[0, 0])),
             math.ones(spatial(x=5, y=5, z=6, a=1))),
            (-math.ones(spatial(x=3, y=4, z=5, a=1)),
             lambda tensor: ConstantExtrapolation(-1).pad(tensor, dict(x=[1, 1], y=[1, 0], z=[0, 1], a=[0, 0])),
             - math.ones(spatial(x=5, y=5, z=6, a=1))),
        ]
        for val_in, func, val_out in test_in_func_out:
            math.assert_close(val_out, func(val_in))
            # TypeError('__bool__ should return bool, returned NotImplementedType')
            # self.assertEqual(val_out, func(val_in))

    def test_pad_2d(self):
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
                # symmetric
                p = math.pad(a, {'x': (2, 2), 'y': (2, 2)}, SYMMETRIC)
                math.assert_close(p.x[2:-2].y[2:-2], a)  # copy inner
                math.assert_close(p.x[2:-2].y[2:-2], a)  # copy inner
                math.assert_close(p.x[2:-2].y[1], a.y[0])
                math.assert_close(p.x[2:-2].y[0], a.y[1])
                math.assert_close(p.x[2:-2].y[-2], a.y[-1])
                math.assert_close(p.x[2:-2].y[-1], a.y[-2])
                # reflect
                p = math.pad(a, {'x': (2, 2), 'y': (2, 2)}, REFLECT)
                math.assert_close(p.x[2:-2].y[2:-2], a)  # copy inner
                math.assert_close(p.x[2:-2].y[1], a.y[1])
                math.assert_close(p.x[2:-2].y[0], a.y[2])
                math.assert_close(p.x[2:-2].y[-2], a.y[-2])
                math.assert_close(p.x[2:-2].y[-1], a.y[-3])
                # mixed
                p = math.pad(a, {'x': (1, 2), 'y': (0, 1)}, combine_sides(x=PERIODIC, y=(ONE, REFLECT)))
                math.print(p)
                self.assertEqual((7, 4, 2), p.shape.sizes)  # dimension check
                math.assert_close(p.x[1:-2].y[:-1], a)  # copy inner
                math.assert_close(p.x[0].y[:-1], a.x[-1])  # periodic
                math.assert_close(p.x[-2:].y[:-1], a.x[:2])  # periodic

    def test_pad_3d(self):
        for t in [
            math.ones(spatial(x=2, y=2, z=2)),
            math.ones(spatial(x=2, y=2, z=2), batch(b1=2)),
            math.ones(spatial(x=2, y=2, z=2), batch(b1=2, b2=2)),
            math.ones(spatial(x=2, y=2, z=2), batch(b1=2, b2=2, b3=2)),
        ]:
            results = []
            for backend in BACKENDS:
                with backend:
                    p = math.pad(t, {i: (1, 1) for i in 'xyz'}, 0)
                    results.append(p)
            math.assert_close(*results)

    def test_pad_4d(self):
        for t in [
            math.ones(spatial(x=2, y=2, z=2, w=2)),
            math.ones(spatial(x=2, y=2, z=2, w=2), batch(b1=2)),
            math.ones(spatial(x=2, y=2, z=2, w=2), batch(b1=2, b2=2)),
            math.ones(spatial(x=2, y=2, z=2, w=2), batch(b1=2, b2=2, b3=2)),
        ]:
            results = []
            for backend in BACKENDS:
                with backend:
                    p = math.pad(t, {i: (1, 1) for i in 'xyzw'}, 0)
                    results.append(p)
            math.assert_close(*results)

    def test_pad_collapsed(self):
        # --- Shapes ---
        a = math.zeros(spatial(b=2, x=10, y=10) & batch(batch=10))
        p = math.pad(a, {'x': (1, 2)}, ZERO)
        self.assertEqual(0, p._native_shape.rank)
        self.assertEqual((10, 2, 13, 10), p.shape.sizes)
        p = math.pad(a, {'x': (1, 2)}, PERIODIC)
        self.assertEqual(0, p._native_shape.rank)
        self.assertEqual((10, 2, 13, 10), p.shape.sizes)
        # --- 1D ---
        p = math.pad(math.ones(spatial(x=3)), {'x': (1, 1)}, 0)
        math.assert_close([0, 1, 1, 1, 0], p)

    def test_pad_negative(self):
        a = math.meshgrid(x=4, y=3)
        p = math.pad(a, {'x': (-1, -1), 'y': (1, -1)}, ZERO)
        math.assert_close(p.y[1:], a.x[1:-1].y[:-1])

    def test_serialize_mixed(self):
        e = combine_sides(x=PERIODIC, y=(ONE, BOUNDARY))
        serialized = e.to_dict()
        e_ = from_dict(serialized)
        self.assertEqual(e, e_)

    def test_normal_tangential(self):
        t = math.ones(spatial(x=2, y=2), channel(vector='x,y'))
        ext = combine_by_direction(normal=ZERO, tangential=-ONE)
        p = math.pad(t, {'x': (1, 1), 'y': (1, 1)}, ext)
        math.assert_close(1, p.x[1:-1].y[1:-1])  # inner
        # x component
        math.assert_close(0, p.vector['x'].x[0].y[1:-1])
        math.assert_close(0, p.vector['x'].x[-1].y[1:-1])
        math.assert_close(-1, p.vector['x'].x[1:-1].y[0])
        math.assert_close(-1, p.vector['x'].x[1:-1].y[-1])
        # y component
        math.assert_close(-1, p.vector['y'].x[0].y[1:-1])
        math.assert_close(-1, p.vector['y'].x[-1].y[1:-1])
        math.assert_close(0, p.vector['y'].x[1:-1].y[0])
        math.assert_close(0, p.vector['y'].x[1:-1].y[-1])
        math.print(p)

    def test_normal_tangential_math(self):
        ext = combine_by_direction(normal=ONE, tangential=PERIODIC)
        self.assertEqual(combine_by_direction(normal=ZERO, tangential=ZERO), ext * ZERO)
        self.assertEqual(ext, ext + ZERO)
        self.assertEqual(ext, ext - ZERO)
        self.assertEqual(ext, ext * ONE)
        self.assertEqual(ext, ext / ONE)
        self.assertEqual(ext, ZERO + ext)
        self.assertEqual(-ext, ZERO - ext)
        self.assertEqual(ext, ONE * ext)
        self.assertEqual(ZERO, ZERO / ext)
        self.assertEqual(ZERO, ZERO * ext)
        self.assertEqual(ZERO, ext * ZERO)
        self.assertEqual(ext, abs(ext))

    def test_symmetric_gradient(self):
        t = wrap([0, 1, 2, 3, 3, 3], spatial('x'))
        padded = SYMMETRIC_GRADIENT.pad(t, {'x': (4,  2)})
        padded_rev = SYMMETRIC_GRADIENT.pad(t.x[::-1], {'x': (2,  4)}).x[::-1]
        math.assert_close([-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 3, 3], padded, padded_rev)

    def test_map(self):
        ext = combine_by_direction(normal=ONE, tangential=PERIODIC)
        self.assertEqual(ext, extrapolation.map(lambda e: e, ext))
        ext = combine_sides(x=PERIODIC, y=(ONE, BOUNDARY))
        self.assertEqual(ext, extrapolation.map(lambda e: e, ext))

    def test_slice_normal_tangential(self):
        INFLOW_LEFT = combine_by_direction(normal=1, tangential=0)
        ext = combine_sides(x=(INFLOW_LEFT, BOUNDARY), y=0)
        self.assertEqual(combine_sides(x=(1, BOUNDARY), y=0), ext[{'vector': 'x'}])
        self.assertEqual(combine_sides(x=(0, BOUNDARY), y=0), ext[{'vector': 'y'}])

    def test_shapes(self):
        self.assertEqual(EMPTY_SHAPE, ONE.shape)
        self.assertEqual(EMPTY_SHAPE, PERIODIC.shape)
        self.assertEqual(EMPTY_SHAPE, BOUNDARY.shape)
        self.assertEqual(EMPTY_SHAPE, SYMMETRIC.shape)
        self.assertEqual(EMPTY_SHAPE, REFLECT.shape)
        v = math.vec(x=1, y=0)
        self.assertEqual(v.shape, shape(ZERO + v))
        self.assertEqual(v.shape, shape(combine_sides(x=v, y=0)))
        self.assertEqual(v.shape, shape(combine_by_direction(normal=v, tangential=0)))

    def test_as_extrapolation(self):
        self.assertEqual(PERIODIC, as_extrapolation('periodic'))
        self.assertEqual(ONE, as_extrapolation('one'))
        self.assertEqual(ZERO, as_extrapolation('zero'))
        self.assertEqual(combine_by_direction(ZERO, 1), as_extrapolation({'normal': 0, 'tangential': 1}))
        self.assertEqual(combine_sides(x=1, y=ZERO_GRADIENT), as_extrapolation({'x': wrap(1), 'y': 'zero-gradient'}))
