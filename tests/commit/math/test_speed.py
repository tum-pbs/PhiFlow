import time
from unittest import TestCase
import numpy as np

from phi import math
from phi.math import spatial, batch, channel


def rnpv(size=64, d=2):
    """ Random NumPy Velocity Tensor """
    return np.random.randn(1, *[size] * d, d).astype(np.float32)


def _assert_equally_fast(f1, f2, n=100, tolerance_per_round=0.001):
    start = time.perf_counter()
    for _ in range(n):
        f1()
    np_time = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(n):
        f2()
    t_time = time.perf_counter() - start
    print(np_time, t_time)
    assert abs(t_time - np_time) / n <= tolerance_per_round


class TestSpeed(TestCase):

    def test_np_speed_op2(self):
        print()
        np1, np2 = rnpv(64), rnpv(64)
        t1 = math.tensor(np1, batch('batch'), spatial('x, y'), channel('vector'))
        t2 = math.tensor(np2, batch('batch'), spatial('x, y'), channel('vector'))
        _assert_equally_fast(lambda: np1 + np2, lambda: t1 + t2, n=10000)
        np1, np2 = rnpv(256), rnpv(256)
        t1 = math.tensor(np1, batch('batch'), spatial('x, y'), channel('vector'))
        t2 = math.tensor(np2, batch('batch'), spatial('x, y'), channel('vector'))
        _assert_equally_fast(lambda: np1 + np2, lambda: t1 + t2, n=1000)

    def test_np_speed_sum(self):
        print()
        np1, np2 = rnpv(64), rnpv(256)
        t1 = math.tensor(np1, batch('batch'), spatial('x, y'), channel('vector'))
        t2 = math.tensor(np2, batch('batch'), spatial('x, y'), channel('vector'))
        _assert_equally_fast(lambda: np.sum(np1), lambda: math.sum(t1, dim=t1.shape), n=10000)
        _assert_equally_fast(lambda: np.sum(np2), lambda: math.sum(t2, dim=t1.shape), n=10000)
