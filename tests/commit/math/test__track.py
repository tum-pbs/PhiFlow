from unittest import TestCase

from phi import math
from phi.math import *
from phi.math._tensors import NativeTensor
from phi.math._track import lin_placeholder, ShiftLinOp


class TestTensors(TestCase):

    def test_linear_operator(self):
        GLOBAL_AXIS_ORDER.x_last()
        direct = math.random_normal(batch=3, x=4, y=3)  # , vector=2
        op = lin_placeholder(direct)

        def linear_function(val):
            val = -val
            val *= 2
            val = math.pad(val, {'x': (2, 0), 'y': (0, 1)}, extrapolation.PERIODIC)
            val = val.x[:-2].y[1:] + val.x[2:].y[:-1]
            val = math.pad(val, {'x': (0, 0), 'y': (0, 1)}, extrapolation.ZERO)
            val = math.pad(val, {'x': (2, 2), 'y': (0, 1)}, extrapolation.BOUNDARY)
            # sl = sl.vector[0]
            return val
            val = val.x[1:4].y[:2]
            return math.sum([val, sl], axis=0) - sl

        functions = [
            linear_function,
            lambda val: math.gradient(val, difference='forward', padding=extrapolation.ZERO, dims='x').gradient[0],
            lambda val: math.gradient(val, difference='backward', padding=extrapolation.PERIODIC, dims='x').gradient[0],
            lambda val: math.gradient(val, difference='central', padding=extrapolation.BOUNDARY, dims='x').gradient[0],
        ]

        for f in functions:
            direct_result = f(direct)
            # print(direct_result.batch[0], 'Direct result')
            op_result = f(op)
            # print(op_result.build_sparse_coordinate_matrix().todense())
            self.assertIsInstance(op_result, ShiftLinOp)
            op_result = NativeTensor(op_result.native(), op_result.shape)
            # print(op_result.batch[0], 'Placeholder result')
            math.assert_close(direct_result, op_result)
