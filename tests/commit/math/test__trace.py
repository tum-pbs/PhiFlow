from unittest import TestCase

from phi import math
from phi.math import *
from phi.math._tensors import NativeTensor
from phi.math._trace import ShiftLinTracer


class TestTensors(TestCase):

    def test_linear_operator(self):
        GLOBAL_AXIS_ORDER.x_last()
        x = math.random_normal(batch=3, x=4, y=3)  # , vector=2

        def linear_function(val):
            val = -val
            val *= 2
            val = math.pad(val, {'x': (2, 0), 'y': (0, 1)}, extrapolation.PERIODIC)
            val = val.x[:-2].y[1:] + val.x[2:].y[:-1]
            val = math.pad(val, {'x': (0, 0), 'y': (0, 1)}, extrapolation.ZERO)
            val = math.pad(val, {'x': (2, 2), 'y': (0, 1)}, extrapolation.BOUNDARY)
            return math.sum([val, val], dim=0) - val

        functions = [
            linear_function,
            lambda val: math.spatial_gradient(val, difference='forward', padding=extrapolation.ZERO, dims='x').gradient[0],
            lambda val: math.spatial_gradient(val, difference='backward', padding=extrapolation.PERIODIC, dims='x').gradient[0],
            lambda val: math.spatial_gradient(val, difference='central', padding=extrapolation.BOUNDARY, dims='x').gradient[0],
        ]
        jit_functions = [math.jit_compile_linear(f) for f in functions]

        for f, jit_f in zip(functions, jit_functions):
            direct_result = f(x)
            jit_result = jit_f(x)
            math.assert_close(direct_result, jit_result)
