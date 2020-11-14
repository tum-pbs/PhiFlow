from unittest import TestCase

import numpy as np

from phi import math
from phi.math._shape import CHANNEL_DIM, BATCH_DIM, shape_stack, shape
from phi.math._tensors import TensorStack, CollapsedTensor


class TestTensors(TestCase):

    def test_tensor_creation(self):
        v = math.tensor(np.ones([1, 4, 3, 2]))
        self.assertEqual((4, 3, 2), v.shape.sizes)
        v = math.tensor(np.ones([10, 4, 3, 2]))
        self.assertEqual((10, 4, 3, 2), v.shape.sizes)
        scalar = math.tensor(np.ones([1, 4, 3, 1]))
        self.assertEqual((4, 3), scalar.shape.sizes)
        a = math.tensor([1, 2, 3])
        self.assertEqual((3,), a.shape.sizes)

    def test_tensor_creation_dims(self):
        a = math.tensor(math.zeros(a=2, b=2), names=':,vector')
        self.assertEqual(('a', 'vector'), a.shape.names)
        self.assertEqual(('spatial', 'spatial'), a.shape.types)

    def test_native_constant_ops(self):
        v = math.tensor(np.ones([1, 4, 3, 2]))
        math.assert_close(v + 1, 2)
        math.assert_close(v * 3, 3)
        math.assert_close(v / 2, 0.5)
        math.assert_close(v ** 2, 1)
        math.assert_close(2 ** v, 2)
        math.assert_close(v + [0, 1], [1, 2])

    def test_native_native_ops(self):
        v = math.ones(batch=2, x=4, y=3, vector=2)
        d = v.unstack('vector')[0]
        math.assert_close(v + d, d + v, 2)
        math.assert_close(v * d, d * v, 1)

    def test_native_unstack(self):
        v = math.ones(batch=10, x=4, y=3, vector=2)
        vx, vy = v.vector.unstack()
        self.assertEqual('(batch=10, x=4, y=3)', repr(vx.shape))
        self.assertEqual(4, len(v.x.unstack()))
        self.assertEqual(10, len(v.batch.unstack()))

    def test_native_slice(self):
        v = math.ones(batch=2, x=4, y=3, vector=2)
        self.assertEqual('(batch=2, x=4, y=3)', repr(v.vector[0].shape))
        self.assertEqual('(batch=2, y=2, vector=2)', repr(v.y[0:2].x[0].shape))

    def test_stacked_shapes(self):
        t0 = math.ones(batch=10, x=4, y=3, vector=2)
        for dim in t0.shape.names:
            tensors = t0.unstack(dim)
            stacked = TensorStack(tensors, dim, t0.shape.get_type(dim))
            self.assertEqual(set(t0.shape.names), set(stacked.shape.names))
            self.assertEqual(t0.shape.volume, stacked.shape.volume)

    def test_stacked_native(self):
        t0 = math.ones(batch=10, x=4, y=3, vector=2)
        tensors = t0.unstack('vector')
        stacked = TensorStack(tensors, 'vector2', CHANNEL_DIM)
        math.assert_close(stacked, t0)
        self.assertEqual((10, 4, 3, 2), stacked.native().shape)
        self.assertEqual((4, 3, 2, 10), stacked.native(order=('x', 'y', 'vector2', 'batch')).shape)
        self.assertEqual((2, 10, 3, 4), stacked.native(order=('vector2', 'batch', 'y', 'x')).shape)  # this should re-stack since only the stacked dimension position is different

    def test_stacked_get(self):
        t0 = math.ones(batch=10, x=4, y=3, vector=2)
        tensors = t0.unstack('vector')
        stacked = TensorStack(tensors, 'channel', CHANNEL_DIM)
        self.assertEqual(tensors, stacked.channel.unstack())
        assert tensors[0] is stacked.channel[0]
        assert tensors[1] is stacked.channel[1:2].channel.unstack()[0]
        self.assertEqual(4, len(stacked.x.unstack()))

    def test_shape_math(self):
        vector = math.ones(x=4, y=3, vector=2)
        vector *= vector.shape.spatial
        math.assert_close(vector.vector[0], 4)
        math.assert_close(vector.vector[1], 3)

    def test_collapsed(self):
        scalar = math.zeros(x=4, y=3)
        math.assert_close(scalar, 0)
        self.assertEqual('(x=4, y=3)', repr(scalar.shape))
        self.assertEqual('(x=4)', repr(scalar.y[0].shape))
        self.assertEqual('()', repr(scalar.y[0].x[0].shape))
        self.assertEqual(3, len(scalar.y.unstack()))

    def test_collapsed_op2(self):
        # Collapsed + Collapsed
        a = math.zeros(vector=4)
        b = math.ones(batch=3)
        c = a + b
        self.assertIsInstance(c, CollapsedTensor)
        self.assertEqual(c.shape.volume, 12)
        self.assertEqual(c.tensor.shape.volume, 1)
        # Collapsed + Native
        n = math.ones(vector=3) + (0, 1, 2)
        math.assert_close(n, (1, 2, 3))

    def test_semi_collapsed(self):
        scalar = math.ones(x=4, y=3)
        scalar = CollapsedTensor(scalar, scalar.shape.expand(10, 'batch', BATCH_DIM))
        self.assertEqual('(batch=10, x=4, y=3)', repr(scalar.shape))
        self.assertEqual(4, len(scalar.x.unstack()))
        self.assertEqual(10, len(scalar.batch.unstack()))
        self.assertEqual('()', repr(scalar.y[0].batch[0].x[0].shape))

    def test_zeros_nonuniform(self):
        nonuniform = shape_stack('stack', BATCH_DIM, shape(time=1, x=3, y=3), shape(x=3, y=4), shape())
        self.assertEqual(math.zeros(nonuniform).shape, nonuniform)
        self.assertEqual(math.ones(nonuniform).shape, nonuniform)
        self.assertEqual(math.random_normal(nonuniform).shape, nonuniform)
        self.assertEqual(math.random_uniform(nonuniform).shape, nonuniform)
