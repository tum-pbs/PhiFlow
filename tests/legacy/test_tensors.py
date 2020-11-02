from unittest import TestCase
from phi.flow import *
from phi.math import *
from phi.math._shape import CHANNEL_DIM, BATCH_DIM
from phi.math._tensors import TensorStack, CollapsedTensor, NativeTensor

import numpy as np

from phi.math._track import lin_placeholder, SparseLinearOperation, ShiftLinOp


class TestTensors(TestCase):

    def test_define_shapes(self):
        self.assertEqual('(channel=2)', repr(define_shape(2)))
        self.assertEqual('(channel0=1, channel1=2, channel2=3)', repr(define_shape((1, 2, 3))))
        self.assertEqual('(batch=10)', repr(define_shape(batch=10)))
        self.assertEqual('(batch=10, time=5)', repr(define_shape(batch={'batch': 10, 'time': 5})))
        self.assertEqual('(batch=10, channel0=2, channel1=1)', repr(define_shape((2, 1), 10)))
        GLOBAL_AXIS_ORDER.x_first()
        self.assertEqual('(y=4, z=5, x=3)', repr(define_shape(y=4, z=5, x=3)))
        self.assertEqual('(batch=10, z=5, x=3, y=4, vector=3)', repr(define_shape(3, z=5, x=3, batch=10, y=4)))
        self.assertEqual((10, 4, 5, 3, 3), define_shape(3, y=4, z=5, x=3, batch=10).sizes)
        self.assertEqual(('batch', 'y', 'z', 'x', 'vector'), define_shape(3, y=4, z=5, x=3, batch=10).names)

    def test_subshape(self):
        shape = define_shape(2, batch=10, x=4, y=3)
        self.assertEqual('(x=4, y=3)', repr(shape.select('x', 'y')))
