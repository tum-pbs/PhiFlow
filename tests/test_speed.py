import time
from unittest import TestCase
import numpy as np

from phi import math


def rnpv(size=64, d=2):
    """ Random NumPy Velocity Tensor """
    return np.random.randn(1, *[size] * d, d)


def _assert_equally_fast(f1, f2, n=100, tolerance_per_round=0.001):
    start = time.time()
    for _ in range(n):
        f1()
    np_time = time.time() - start
    start = time.time()
    for _ in range(n):
        f2()
    t_time = time.time() - start
    print(np_time, t_time)
    assert abs(t_time - np_time) / n <= tolerance_per_round


class TestMath(TestCase):

    def test_speed_op2(self):
        np1, np2 = rnpv(64), rnpv(64)
        t1, t2 = math.tensor(np1, np2)
        _assert_equally_fast(lambda: np1 + np2, lambda: t1 + t2, n=10000)
        np1, np2 = rnpv(256), rnpv(256)
        t1, t2 = math.tensor(np1, np2)
        _assert_equally_fast(lambda: np1 + np2, lambda: t1 + t2, n=1000)

    def test_speed_sum(self):
        np1, np2 = rnpv(64), rnpv(256)
        t1, t2 = math.tensor(np1, np2)
        _assert_equally_fast(lambda: np.sum(np1), lambda: math.sum(t1), n=10000)
        _assert_equally_fast(lambda: np.sum(np2), lambda: math.sum(t2), n=10000)
