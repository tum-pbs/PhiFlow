from unittest import TestCase

import numpy as np

import phi
from phi import math
from phi.math import channel, batch
from phi.math._shape import CHANNEL_DIM, BATCH_DIM, shape_stack, spatial
from phi.math._tensors import TensorStack, CollapsedTensor, wrap, tensor
from phi.math.backend import Backend

BACKENDS = phi.detect_backends()


class TestTensors(TestCase):

    def test_tensor_from_constant(self):
        for backend in BACKENDS:
            with backend:
                for const in (1, 1.5, True, 1+1j):
                    tens = math.wrap(const)
                    self.assertEqual(math.NUMPY, tens.default_backend)
                    self.assertTrue(isinstance(tens.native(), (int, float, bool, complex)), msg=backend)
                    math.assert_close(tens, const)
                    tens = math.tensor(const)
                    self.assertEqual(backend, math.choose_backend(tens), f'{const} was not converted to the specified backend')
                    math.assert_close(tens, const)

    def test_tensor_from_native(self):
        for creation_backend in BACKENDS:
            native = creation_backend.ones((4,))
            for backend in BACKENDS:
                with backend:
                    tens = math.tensor(native, convert=False)
                    self.assertEqual(creation_backend, tens.default_backend)
                    math.assert_close(tens, native)
                    tens = math.tensor(native)
                    self.assertEqual(backend, tens.default_backend, f'Conversion failed from {creation_backend} to {backend}')
                    math.assert_close(tens, native)

    def test_tensor_from_tuple_of_numbers(self):
        data_tuple = (1, 2, 3)
        for backend in BACKENDS:
            with backend:
                tens = math.tensor(data_tuple, convert=False)
                self.assertEqual(math.NUMPY, math.choose_backend(tens))
                math.assert_close(tens, data_tuple)
                tens = math.tensor(data_tuple)
                self.assertEqual(backend, math.choose_backend(tens))
                math.assert_close(tens, data_tuple)

    def test_tensor_from_tuple_of_tensor_like(self):
        native = ([1, 2, 3], math.zeros(channel(vector=3)))
        for backend in BACKENDS:
            with backend:
                tens = wrap(native, batch(stack=2), channel(vector=3))
                self.assertEqual(math.NUMPY, math.choose_backend(tens))
                self.assertEqual(batch(stack=2) & channel(vector=3), tens.shape)
                tens = tensor(native, batch(stack=2), channel(vector=3))
                self.assertEqual(backend, math.choose_backend(tens))
                self.assertEqual(batch(stack=2) & channel(vector=3), tens.shape)

    def test_tensor_from_tensor(self):
        ref = math.stack([math.zeros(spatial(x=5)), math.zeros(spatial(x=4))], batch('stack'))
        for backend in BACKENDS:
            with backend:
                tens = math.tensor(ref, convert=False)
                self.assertEqual(math.NUMPY, math.choose_backend(tens))
                self.assertEqual(2, tens.shape.get_size('stack'))
                self.assertEqual(('stack', 'x'), tens.shape.names)
                tens = math.tensor(ref)
                self.assertEqual(backend, math.choose_backend(tens))
                self.assertEqual(backend, math.choose_backend(tens.stack[0]))
                self.assertEqual(backend, math.choose_backend(tens.stack[1]))
                tens = math.tensor(ref, batch('n1', 'n2'))
                self.assertEqual(backend, math.choose_backend(tens))

    def test_multi_dim_tensor_from_numpy(self):
        v = math.tensor(np.ones([1, 4, 3, 2]), batch('batch'), spatial('x,y'), channel('vector'))
        self.assertEqual((1, 4, 3, 2), v.shape.sizes)
        v = math.tensor(np.ones([10, 4, 3, 2]), batch('batch'), spatial('x,y'), channel('vector'))
        self.assertEqual((10, 4, 3, 2), v.shape.sizes)

    def test_native_constant_ops(self):
        v = math.tensor(np.ones([1, 4, 3, 2]), batch('batch'), spatial('x,y'), channel('vector'))
        math.assert_close(v + 1, 2)
        math.assert_close(v * 3, 3)
        math.assert_close(v / 2, 0.5)
        math.assert_close(v ** 2, 1)
        math.assert_close(2 ** v, 2)
        math.assert_close(v + [0, 1], [1, 2])

    def test_native_native_ops(self):
        v = math.ones(batch(batch=10) & spatial(x=4, y=3) & channel(vector=2))
        d = v.unstack('vector')[0]
        math.assert_close(v + d, d + v, 2)
        math.assert_close(v * d, d * v, 1)

    def test_native_unstack(self):
        v = math.ones(batch(batch=10), spatial(x=4, y=3), channel(vector=2))
        vx, vy = v.vector.unstack()
        self.assertEqual((10, 4, 3), vx.shape.sizes)
        self.assertEqual(4, len(v.x.unstack()))
        self.assertEqual(10, len(v.batch.unstack()))

    def test_native_slice(self):
        v = math.ones(batch(batch=10), spatial(x=4, y=3), channel(vector=2))
        self.assertEqual((10, 4, 3), v.vector[0].shape.sizes)
        self.assertEqual((10, 2, 2), v.y[0:2].x[0].shape.sizes)

    def test_stacked_shapes(self):
        t0 = math.ones(batch(batch=10) & spatial(x=4, y=3) & channel(vector=2))
        for dim in t0.shape.names:
            tensors = t0.unstack(dim)
            stacked = math.stack(tensors, t0.shape[dim].with_sizes([None]))
            self.assertEqual(set(t0.shape.names), set(stacked.shape.names))
            self.assertEqual(t0.shape.volume, stacked.shape.volume)

    def test_stacked_native(self):
        t0 = math.ones(batch(batch=10) & spatial(x=4, y=3) & channel(vector=2))
        tensors = t0.unstack('vector')
        stacked = math.stack(tensors, channel('vector2'))
        math.assert_close(stacked, t0)
        self.assertEqual((10, 4, 3, 2), stacked.native(stacked.shape).shape)
        self.assertEqual((4, 3, 2, 10), stacked.native(order=('x', 'y', 'vector2', 'batch')).shape)
        self.assertEqual((2, 10, 3, 4), stacked.native(order=('vector2', 'batch', 'y', 'x')).shape)  # this should re-stack since only the stacked dimension position is different

    def test_stacked_get(self):
        t0 = math.ones(batch(batch=10) & spatial(x=4, y=3) & channel(vector=2))
        tensors = t0.unstack('vector')
        stacked = math.stack(tensors, channel('channel'))
        self.assertEqual(tensors, stacked.channel.unstack())
        assert tensors[0] is stacked.channel[0]
        assert tensors[1] is stacked.channel[1:2].channel.unstack()[0]
        self.assertEqual(4, len(stacked.x.unstack()))

    def test_shape_math(self):
        vector = math.ones(spatial(x=4, y=3) & channel(vector=2))
        vector *= vector.shape.spatial
        math.assert_close(vector.vector[0], 4)
        math.assert_close(vector.vector[1], 3)

    def test_collapsed(self):
        scalar = math.zeros(spatial(x=4, y=3))
        math.assert_close(scalar, 0)
        self.assertEqual((4, 3), scalar.shape.sizes)
        self.assertEqual(4, scalar.y[0].shape.size)
        self.assertEqual(0, scalar.y[0].x[0].shape.rank)
        self.assertEqual(3, len(scalar.y.unstack()))

    def test_collapsed_op2(self):
        # Collapsed + Collapsed
        a = math.zeros(channel(vector=4))
        b = math.ones(batch(batch=3))
        c = a + b
        self.assertIsInstance(c, CollapsedTensor)
        self.assertEqual(c.shape.volume, 12)
        self.assertEqual(c._inner.shape.volume, 1)
        # Collapsed + Native
        n = math.ones(channel(vector=3)) + (0, 1, 2)
        math.assert_close(n, (1, 2, 3))

    def test_semi_collapsed(self):
        scalar = math.ones(spatial(x=4, y=3))
        scalar = CollapsedTensor(scalar, scalar.shape._expand(batch(batch=10)))
        self.assertEqual((10, 4, 3), scalar.shape.sizes)
        self.assertEqual(4, len(scalar.x.unstack()))
        self.assertEqual(10, len(scalar.batch.unstack()))
        self.assertEqual(0, scalar.y[0].batch[0].x[0].shape.rank)

    def test_zeros_nonuniform(self):
        nonuniform = shape_stack(batch('stack'), batch(time=1) & spatial(x=3, y=3), spatial(x=3, y=4), channel())
        self.assertEqual(math.zeros(nonuniform).shape, nonuniform)
        self.assertEqual(math.ones(nonuniform).shape, nonuniform)
        self.assertEqual(math.random_normal(nonuniform).shape, nonuniform)
        self.assertEqual(math.random_uniform(nonuniform).shape, nonuniform)

    def test_repr(self):
        print("--- Eager ---")
        print(repr(math.zeros(batch(b=10))))
        print(repr(math.zeros(batch(b=10)) > 0))
        print(repr(math.ones(channel(vector=3))))
        print(repr(math.ones(batch(vector=3))))

        def tracable(x):
            print(x)
            return x

        print("--- Placeholders ---")
        for backend in BACKENDS:
            if backend.supports(Backend.jit_compile):
                with backend:
                    math.jit_compile(tracable)(math.ones(channel(vector=3)))


    def test_tensor_like(self):

        class Success(Exception): pass

        class MyObjV:

            def __init__(self, x):
                self.x = x

            def __value_attrs__(self):
                return 'x',

            def __with_tattrs__(self, **tattrs):
                math.assert_close(tattrs['x'], 1)
                raise Success

        class MyObjT:

            def __init__(self, x1, x2):
                self.x1 = x1
                self.x2 = x2

            def __variable_attrs__(self):
                return 'x1', 'x2'

        v = MyObjV(math.wrap(0))
        t = MyObjT(math.wrap(0), math.wrap(1))
        self.assertIsInstance(v, math.TensorLike)
        self.assertIsInstance(t, math.TensorLike)
        try:
            math.cos(v)
        except Success:
            pass
        try:
            math.cos(t)
        except AssertionError:
            pass

    def test_Dict(self):
        d1 = math.Dict(a=1, b=math.ones(), c=math.ones(spatial(x=3)))
        math.assert_close(d1 * 2, d1 + d1, 2 * d1, 2 / d1)
        math.assert_close(0 + d1, d1, d1 - 0, abs(d1), round(d1))
        math.assert_close(-d1, 0 - d1)
        math.assert_close(d1 // 2, d1 * 0, d1 % 1)
        math.assert_close(d1 / 2, d1 * 0.5, 0.5 * d1)
        math.assert_close(math.sin(d1 * 0), d1 * 0)

    def test_collapsed_non_uniform_tensor(self):
        non_uniform = math.stack([math.zeros(spatial(a=2)), math.ones(spatial(a=3))], batch('b'))
        e = math.expand(non_uniform, channel('vector'))
        assert e.shape.without('vector') == non_uniform.shape
