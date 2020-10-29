from unittest import TestCase
from phi.flow import *
from phi.math import *
from phi.math._shape import CHANNEL_DIM, BATCH_DIM
from phi.math._tensors import TensorStack, CollapsedTensor

import numpy as np

from phi.math._track import as_sparse_linear_operation, SparseLinearOperation


class TestTensors(TestCase):

    def test_define_shapes(self):
        self.assertEqual('(channel=2)', repr(define_shape(2)))
        self.assertEqual('(channel0=1, channel1=2, channel2=3)', repr(define_shape((1, 2, 3))))
        self.assertEqual('(batch=10)', repr(define_shape(batch=10)))
        self.assertEqual('(batch=10, time=5)', repr(define_shape(batch={'batch': 10, 'time': 5})))
        self.assertEqual('(batch=10, channel0=2, channel1=1)', repr(define_shape((2, 1), 10)))
        physics_config.x_first()
        self.assertEqual('(y=4, z=5, x=3)', repr(define_shape(y=4, z=5, x=3)))
        self.assertEqual('(batch=10, z=5, x=3, y=4, vector=3)', repr(define_shape(3, z=5, x=3, batch=10, y=4)))
        self.assertEqual((10, 4, 5, 3, 3), define_shape(3, y=4, z=5, x=3, batch=10).sizes)
        self.assertEqual(('batch', 'y', 'z', 'x', 'vector'), define_shape(3, y=4, z=5, x=3, batch=10).names)

    def test_subshape(self):
        shape = define_shape(2, batch=10, x=4, y=3)
        self.assertEqual('(x=4, y=3)', repr(shape.select('x', 'y')))

    def test_infer_shape(self):
        shape = infer_shape([1, 2, 3, 4], batch_dims=1, channel_dims=1)
        self.assertEqual(2, shape.spatial.rank)
        self.assertEqual(shape, infer_shape([1, 2, 3, 4]))
        shape = infer_shape([1, 2, 3, 4], batch_dims=0, channel_dims=0)
        self.assertEqual(4, shape.spatial.rank)
        shape = infer_shape([1, 2, 3, 4], batch_dims=0, spatial_dims=0)
        self.assertEqual(4, shape.channel.rank)

    def test_tensor_creation(self):
        v = tensor(np.ones([1, 4, 3, 2]))
        self.assertEqual((4, 3, 2), v.shape.sizes)
        v = tensor(np.ones([10, 4, 3, 2]))
        self.assertEqual((10, 4, 3, 2), v.shape.sizes)
        scalar = tensor(np.ones([1, 4, 3, 1]))
        self.assertEqual((4, 3), scalar.shape.sizes)
        a = tensor([1, 2, 3])
        self.assertEqual((3,), a.shape.sizes)

    def test_dimension_types(self):
        physics_config.x_first()
        v = tensor(np.ones([10, 4, 3, 2]))
        self.assertEqual(v.x.index, 1)
        self.assertEqual(v.x.name, 'x')
        self.assertTrue(v.x.is_spatial)
        self.assertTrue(v.batch.is_batch)
        b = v.x.as_batch()
        self.assertTrue(b.x.is_batch)

    def test_native_unstack(self):
        physics_config.x_first()
        v = tensor(np.ones([10, 4, 3, 2]))
        vx, vy = v.vector.unstack()
        self.assertEqual('(batch=10, x=4, y=3)', repr(vx.shape))
        self.assertEqual(4, len(v.x.unstack()))
        self.assertEqual(10, len(v.batch.unstack()))

    def test_native_slice(self):
        physics_config.x_first()
        v = tensor(np.ones([1, 4, 3, 2]))
        self.assertEqual('(x=4, y=3)', repr(v.vector[0].shape))
        self.assertEqual('(y=2, vector=2)', repr(v.y[0:2].x[0].shape))

    def test_native_constant_ops(self):
        v = tensor(np.ones([1, 4, 3, 2]))
        math.assert_close(v + 1, 2)
        math.assert_close(v * 3, 3)
        math.assert_close(v / 2, 0.5)
        math.assert_close(v ** 2, 1)
        math.assert_close(2 ** v, 2)
        math.assert_close(v + [0, 1], [1, 2])

    def test_native_native_ops(self):
        v = tensor(np.ones([1, 4, 3, 2]))
        d = v.unstack('vector')[0]
        math.assert_close(v + d, d + v, 2)
        math.assert_close(v * d, d * v, 1)

    def test_math_functions(self):
        v = tensor(np.ones([1, 4, 3, 2]))
        math.assert_close(math.maximum(0, v), 1)
        math.assert_close(math.maximum(0, -v), 0)

    def test_stacked_shapes(self):
        physics_config.x_last()
        t0 = tensor(np.ones([10, 4, 3, 2]))
        for dim in t0.shape.names:
            tensors = t0.unstack(dim)
            stacked = TensorStack(tensors, dim, t0.shape.get_type(dim))
            self.assertEqual(set(t0.shape.names), set(stacked.shape.names))
            self.assertEqual(t0.shape.volume, stacked.shape.volume)

    def test_stacked_native(self):
        physics_config.x_last()
        t0 = tensor(np.ones([10, 4, 3, 2]))
        tensors = t0.unstack('vector')
        stacked = TensorStack(tensors, 'vector2', CHANNEL_DIM)
        math.assert_close(stacked, t0)
        self.assertEqual((10, 4, 3, 2), stacked.native().shape)
        self.assertEqual((3, 4, 2, 10), stacked.native(order=('x', 'y', 'vector2', 'batch')).shape)
        self.assertEqual((2, 10, 4, 3), stacked.native(order=('vector2', 'batch', 'y', 'x')).shape)  # this should re-stack since only the stacked dimension position is different

    def test_stacked_get(self):
        physics_config.x_first()
        t0 = tensor(np.ones([10, 4, 3, 2]))
        tensors = t0.unstack('vector')
        stacked = TensorStack(tensors, 'channel', CHANNEL_DIM)
        self.assertEqual(tensors, stacked.channel.unstack())
        assert tensors[0] is stacked.channel[0]
        assert tensors[1] is stacked.channel[1:2].channel.unstack()[0]
        self.assertEqual(4, len(stacked.x.unstack()))

    def test_collapsed(self):
        physics_config.x_first()
        scalar = zeros([1, 4, 3, 1])
        math.assert_close(scalar, 0)
        self.assertEqual('(x=4, y=3)', repr(scalar.shape))
        self.assertEqual('(x=4)', repr(scalar.y[0].shape))
        self.assertEqual('()', repr(scalar.y[0].x[0].shape))
        self.assertEqual(3, len(scalar.y.unstack()))

    def test_semi_collapsed(self):
        physics_config.x_first()
        scalar = tensor(np.ones([1, 4, 3, 1]))
        scalar = CollapsedTensor(scalar, scalar.shape.expand(10, 'batch', BATCH_DIM))
        self.assertEqual('(batch=10, x=4, y=3)', repr(scalar.shape))
        self.assertEqual(4, len(scalar.x.unstack()))
        self.assertEqual(10, len(scalar.batch.unstack()))
        self.assertEqual('()', repr(scalar.y[0].batch[0].x[0].shape))

    def test_shape_math(self):
        vector = tensor(np.ones([1, 4, 3, 2]))
        vector *= vector.shape.spatial
        math.assert_close(vector.vector[0], 4)
        math.assert_close(vector.vector[1], 3)

    def test_linear_operator(self):
        GLOBAL_AXIS_ORDER.x_last()
        direct = math.random_normal([10, 4, 3, 1])
        op = as_sparse_linear_operation(direct)

        def linear_function(val):
            val *= 2
            sl = val.x[:3].y[:2]
            val = math.pad(val, {'x': (2, 1), 'y': (1, 2)}, mode=math.extrapolation.ZERO)
            val = val.x[1:4].y[:2]
            return math.sum([val, sl], axis=0) - sl

        direct_result = linear_function(direct)
        print()
        print(direct_result.numpy()[0])
        op_result = linear_function(op)
        print()
        print(op_result.numpy()[0])
        math.assert_close(direct_result, op_result)
        self.assertIsInstance(op_result, SparseLinearOperation)
