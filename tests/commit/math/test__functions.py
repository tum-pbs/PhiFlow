from unittest import TestCase

from phi import math


def assert_not_close(*tensors, rel_tolerance, abs_tolerance):
    try:
        math.assert_close(*tensors, rel_tolerance, abs_tolerance)
        raise BaseException(AssertionError('Values are not close'))
    except AssertionError:
        pass


class TestMathFunctions(TestCase):

    def test_assert_close(self):
        math.assert_close(math.zeros(a=10), math.zeros(a=10), math.zeros(a=10), rel_tolerance=0, abs_tolerance=0)
        assert_not_close(math.zeros(a=10), math.ones(a=10), rel_tolerance=0, abs_tolerance=0)
        for scale in (1, 0.1, 10):
            math.assert_close(math.zeros(a=10), math.ones(a=10) * scale, rel_tolerance=0, abs_tolerance=scale)
            math.assert_close(math.zeros(a=10), math.ones(a=10) * scale, rel_tolerance=1, abs_tolerance=0)
            assert_not_close(math.zeros(a=10), math.ones(a=10) * scale, rel_tolerance=0.9, abs_tolerance=0)
            assert_not_close(math.zeros(a=10), math.ones(a=10) * scale, rel_tolerance=0, abs_tolerance=0.9 * scale)
        math.set_precision(64)
        assert_not_close(math.zeros(a=10), math.ones(a=10) * 1e-100, rel_tolerance=0, abs_tolerance=0)
        math.assert_close(math.zeros(a=10), math.ones(a=10) * 1e-100, rel_tolerance=0, abs_tolerance=1e-15)

    def test_concat(self):
        c = math.concat([math.zeros(b=3, a=2), math.ones(a=2, b=4)], 'b')
        self.assertEqual(2, c.shape.a)
        self.assertEqual(7, c.shape.b)
        math.assert_close(c.b[:3], 0)
        math.assert_close(c.b[3:], 1)
