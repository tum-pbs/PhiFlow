from unittest import TestCase

import numpy
import numpy as np

import phi
from phi import math
from phi.math import channel, batch, DType, vec, stack, expand
from phi.math._shape import CHANNEL_DIM, BATCH_DIM, shape_stack, spatial, instance
from phi.math._tensors import TensorStack, wrap, tensor, cached, disassemble_tensors, assemble_tensors, \
    Layout, equality_by_ref
from phi.math.backend import Backend
from phi.math.magic import PhiTreeNode

BACKENDS = phi.detect_backends()


class TestTensors(TestCase):

    def test_tensor_from_constant(self):
        for backend in BACKENDS:
            with backend:
                for const in (1, 1.5, True, 1+1j):
                    tens = math.wrap(const)
                    self.assertEqual(math.NUMPY, tens.default_backend)
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

    def test_tensor_from_shape(self):
        s = spatial(x=4, y=3)
        t = math.tensor(s)
        math.assert_close(t, [4, 3])
        self.assertEqual(t.shape.get_item_names('dims'), ('x', 'y'))

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
        self.assertEqual(c.shape.volume, 12)
        self.assertEqual(c._native_shape.volume, 1)
        # Collapsed + Native
        n = math.ones(channel(vector=3)) + (0, 1, 2)
        math.assert_close(n, (1, 2, 3))

    def test_semi_collapsed(self):
        scalar = math.ones(spatial(x=4, y=3))
        scalar = expand(scalar, batch(batch=10))
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

    def test_close_different_shapes(self):
        a = math.ones(channel(vector='x,y'))
        b = math.wrap(3)
        self.assertFalse(math.close(a, b))
        self.assertFalse(math.close(cached(a), b))
        math.assert_close(a+2, b)

    def test_repr(self):
        print("--- Eager ---")
        print(repr(math.zeros(batch(b=10))))
        print(repr(math.zeros(batch(b=10)) > 0))
        print(repr(math.ones(channel(vector=3))))
        print(repr(math.ones(channel(vector=3), dtype=DType(int, 64))))
        print(repr(math.ones(channel(vector=3), dtype=DType(float, 64))))
        print(repr(math.ones(batch(vector=3))))
        print(repr(math.random_normal(batch(b=10))))
        print(repr(math.random_normal(batch(b=10), dtype=DType(float, 64)) * 1e-6))

        def tracable(x):
            print(x)
            return x

        print("--- Placeholders ---")
        for backend in BACKENDS:
            if backend.supports(Backend.jit_compile):
                with backend:
                    math.jit_compile(tracable)(math.ones(channel(vector=3)))

    def test_format(self):
        for value in [
            math.random_normal(channel(vector='x,y'), dtype=(float, 32)),
            math.random_normal(channel(vector='x,y'), dtype=(float, 64)),
            math.random_normal(channel(vector='x,y'), dtype=(int, 32)),
            math.random_normal(batch(b=6), channel(vector='x,y'), dtype=(int, 64)),
            math.random_normal(batch(b=6), channel(vector='x,y'), dtype=(float, 64)),
            math.ones(),
            math.ones(spatial(x=4, y=3)),
        ]:
            for layout in [
                'auto',
                'row',
                'numpy',
                'full',
                'summary'
            ]:
                print(f"#  {layout}")
                print(f"{value:{layout}:color:no-shape:no-dtype:threshold=1:.1f}")
                print(f"{value:{layout}:shape:dtype:no-color:threshold=1:.8f}")
                print(f"{value:{layout}}")
                print()

    def test_tensor_like(self):

        class Success(Exception): pass

        class MyObjV:

            def __init__(self, x):
                self.x = x

            def __value_attrs__(self):
                return 'x',

            def __with_attrs__(self, **tattrs):
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
        self.assertIsInstance(v, PhiTreeNode)
        self.assertIsInstance(t, PhiTreeNode)
        try:
            math.cos(v)
        except Success:
            pass
        try:
            math.cos(t)
        except ValueError:
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
        e = math.expand(non_uniform, channel(vector='x,y'))
        assert e.shape.without('vector') == non_uniform.shape

    def test_slice_by_item_name(self):
        t = math.tensor(spatial(x=4, y=3))
        math.assert_close(t.dims['x'], 4)
        math.assert_close(t.dims['y'], 3)
        math.assert_close(t.dims['y,x'], (3, 4))
        math.assert_close(t.dims[('y', 'x')], (3, 4))
        math.assert_close(t.dims[spatial('x,y')], (4, 3))

    def test_slice_by_bool_tensor(self):
        indices = math.meshgrid(x=2, y=2)
        sel = indices.x[wrap((True, False), spatial('x'))]
        self.assertEqual(1, sel.x.size)

    def test_slice_by_int_tensor(self):
        indices = math.meshgrid(x=2, y=2)
        sel = indices.x[wrap((1, 0), spatial('x'))]
        math.assert_close([1, 0], sel.vector['x'].y[0])

    def test_slice_str(self):
        x = math.random_normal(batch(b=2), channel(c='x,y,z'))
        math.assert_close(x.c[0], x['x'])
        math.assert_close(x.c[(0, 1)], x['x,y'])

    def test_slice_int(self):
        x = math.random_normal(batch(b=2), channel(c='x,y,z'))
        math.assert_close(x.c[0], x[0])
        math.assert_close(x.c[(0, 1)], x[[0, 1]])

    def test_serialize_tensor(self):
        t = math.random_normal(batch(batch=10), spatial(x=4, y=3), channel(vector=2))
        math.assert_close(t, math.from_dict(math.to_dict(t)))

    def test_flip_item_names(self):
        t = math.zeros(spatial(x=4, y=3), channel(vector='x,y'))
        self.assertEqual(('x', 'y'), t.vector.item_names)
        t_ = t.vector.flip()
        self.assertEqual(('y', 'x'), t_.vector.item_names)
        t_ = t.vector[::-1]
        self.assertEqual(('y', 'x'), t_.vector.item_names)

    def test_op2_incompatible_item_names(self):
        t1 = math.random_normal(channel(vector='x,y,z'))
        t2 = math.random_normal(channel(vector='r,g,b'))
        self.assertEqual(('r', 'g', 'b'), t2.vector.item_names)
        try:
            t1 + t2
            self.fail("Tensors with incompatible item names cannot be added")
        except math.IncompatibleShapes:
            pass
        t1 + t1
        t2_ = t2 + math.random_normal(channel(vector=3))
        self.assertEqual(('r', 'g', 'b'), t2_.vector.item_names)
        t2_ = math.random_normal(channel(vector=3)) + t2
        self.assertEqual(('r', 'g', 'b'), t2_.vector.item_names)

    def test_layout_single(self):
        a = object()
        t = math.layout(a)
        self.assertEqual(a, t.native())

    def test_layout_list(self):
        a = ['a', 'b', 'c']
        t = math.layout(a, channel(letters=a))
        self.assertEqual(a, t.native())
        self.assertEqual('a', t.letters['a'].native())
        self.assertEqual('a', t.letters['b, a'].letters['a'].native())

    def test_layout_tree(self):
        a = [['a', 'b1'], 'b2', 'c']
        t = math.layout(a, channel(outer='list,b2,c', inner=None))
        self.assertEqual(a, t.native())
        self.assertEqual(['a', 'b1'], t.outer['list'].native())
        self.assertEqual('a', t.outer['list'].inner[0].native())
        self.assertEqual(['a', 'b', 'c'], t.inner[0].native())
        self.assertEqual('a', t.inner[0].outer['list'].native())

    def test_layout_size(self):
        a = [['a', 'b1'], 'b2', 'c']
        t = math.layout(a, channel(outer='list,b2,c', inner=None))
        self.assertEqual(3, t.shape.get_size('outer'))
        self.assertEqual(2, t.outer['list'].shape.get_size('inner'))
        self.assertEqual(1, t.outer['c'].shape.get_size('inner'))

    def test_layout_dict(self):
        a = {'a': 'text', 'b': [0, 1]}
        t = math.layout(a, channel('dict,inner'))
        self.assertEqual(a, t.native())
        self.assertEqual(('a', 'b'), t.shape.get_item_names('dict'))
        self.assertEqual(a, t.native())
        self.assertEqual('text', t.dict['a'].native())
        self.assertEqual('e', t.dict['a'].inner[1].native())
        self.assertEqual(1, t.dict['b'].inner[1].native())
        self.assertEqual(('e', 1), t.inner[1].native())

    def test_layout_dict_conflict(self):
        a = [dict(a=1), dict(b=2)]
        t = math.layout(a, channel('outer,dict'))
        self.assertEqual(None, t.shape.get_item_names('dict'))
        self.assertEqual(a, t.native())
        self.assertEqual([1, 2], t.dict[0].native())
        self.assertEqual(2, t.dict[0].outer[1].native())

    def test_layout_None(self):
        none = math.layout(None)
        self.assertEqual(None, none.native())
        l = math.layout([None, None], channel('v'))
        self.assertEqual(None, none.v[0].native())

    def test_iterate_0d(self):
        total = 0.
        for value in math.ones():
            total += value
        self.assertIsInstance(total, float)
        self.assertEqual(total, 1)

    def test_iterate_1d(self):
        total = 0.
        for value in math.ones(channel(vector=3)):
            total += value
        self.assertIsInstance(total, float)
        self.assertEqual(total, 3)

    def test_iterate_2d(self):
        total = 0.
        for value in math.ones(channel(v1=2, v2=2)):
            total += value
        self.assertIsInstance(total, float)
        self.assertEqual(total, 4)

    def test_iterate_layout(self):
        a = [dict(a=1), dict(b=2)]
        t = math.layout(a, channel('outer,dict'))
        total = []
        for d in t:
            total.append(d)
        self.assertEqual(total, [1, 2])

    def test_default_backend_layout(self):
        self.assertIsNone(math.layout(None).default_backend)

    def test_reduction_properties(self):
        for backend in BACKENDS:
            with backend:
                t = math.meshgrid(x=2, y=2)
                self.assertEqual(0.5, t.mean)
                self.assertEqual(0.5, t.std)
                self.assertEqual(1, t.max)
                self.assertEqual(0, t.min)
                self.assertEqual(4, t.sum)
                self.assertEqual(False, t.all)
                self.assertEqual(True, t.any)

    def test_nested_reduce(self):
        t = math.expand(stack([math.ones(spatial(x=4, y=3)), -math.ones(spatial(x=4, y=3))], channel(vector='x,y')), batch(b=10))
        self.assertEqual(0, t.mean)
        self.assertEqual(1, t.std)
        self.assertEqual(-1, t.min)
        self.assertEqual(1, t.max)
        self.assertEqual(0, t.sum)
        self.assertEqual(True, t.all)
        self.assertEqual(True, t.any)

    def test_iter_dim(self):
        slices = tuple(math.zeros(channel(vector='x,y')).vector)
        self.assertEqual(2, len(slices))

    def test_matmul_reduce(self):
        a = math.meshgrid(x=4, y=3)
        math.assert_close(a.y[0], a.y * (1, 0, 0))
        math.assert_close(a.y[1], a.y * (0, 1, 0))

    def test_zero_dim(self):
        nothing = math.random_uniform(spatial(x=5)).x[:0]
        self.assertEqual(spatial(x=0), nothing.shape)
        nothing = math.zeros(spatial(x=5)).x[:0]
        self.assertEqual(spatial(x=0), nothing.shape)

    def test_disassemble_assemble(self):
        for t in [
            math.zeros(batch(b=2, c=2)),
            math.ones(batch(b=10)) * wrap((1, 2), channel('vector')),
        ]:
            natives, shapes, specs = disassemble_tensors([t], expand=False)
            restored = assemble_tensors(natives, specs)
            math.assert_close(t, restored[0])
            print(restored)

    def test_is_number(self):
        self.assertTrue(math.is_scalar(1.))
        self.assertTrue(math.is_scalar(-3))
        self.assertTrue(math.is_scalar(True))
        self.assertTrue(math.is_scalar(1j))
        self.assertTrue(math.is_scalar(math.zeros()))
        self.assertTrue(math.is_scalar(numpy.zeros(())))
        self.assertFalse(math.is_scalar(math.zeros(spatial(x=4))))
        self.assertFalse(math.is_scalar(numpy.zeros((1,))))

    def test_compatible_tensor(self):
        a = wrap([1, 2, 3], instance('points')) * vec(x=0, y=1)
        self.assertEqual(instance(points=3) & channel(vector='x,y'), a.shape)

    def test_numpy_cast(self):
        for t in [math.zeros(spatial(x=4)), math.random_uniform(spatial(x=4))]:
            self.assertEqual(numpy.float32, t.numpy().dtype)
            with math.precision(64):
                self.assertEqual(numpy.float64, t.numpy().dtype)

    def test_change_dims(self):
        t = math.expand(math.random_normal(spatial(x=4)), batch(b=10))
        self.assertEqual(batch(b=10, a=4), t.x.as_batch('a').shape)
        self.assertEqual(t.shape, t.nodim.as_batch('a').shape)
        self.assertEqual(('b', '~x'), t.x.as_dual().shape.names)
        self.assertEqual(('b', 'x'), t.x.as_dual().x.dual.as_spatial().shape.names)
        self.assertEqual(('b', 'x'), t.x.as_dual().b.x.dual.as_spatial().shape.names)

    def test_device(self):
        for backend in BACKENDS:
            with backend:
                t = math.zeros()
                self.assertEqual(t.device, t.default_backend.get_default_device())
                self.assertEqual(backend.get_default_device(), t.device)

    def test_wrap_layout(self):
        for a in ['abc', ('a', 'b', 'c'), math.layout('a')]:
            t = wrap(a)
            self.assertIsInstance(t, Layout, type(t))

    def test_layout_shape_completion(self):
        l = math.layout({'a': [1, 2], 'b': (0, 1)}, channel('letter,index'))
        self.assertEqual(channel(letter='a,b', index=2), l.shape, l.shape)

    def test_layout_ops(self):
        l = math.layout({'a': [1, 2], 'b': (0, 1)}, channel('letter,index'))
        math.assert_close(2 * l, l * 2, l + l)

    def test_layout_assert_close(self):
        l = math.layout({'a': [1, 2], 'b': (0, 1)}, channel('letter,index'))
        math.assert_close(l, l)
        math.assert_close(0, l * 0)
        try:
            math.assert_close(0, l)
            raise RuntimeError("Faulty assertion passed")
        except AssertionError:
            pass
        try:
            math.assert_close(l, l + 1)
            raise RuntimeError("Faulty assertion passed")
        except AssertionError:
            pass

    def test_layout_nested_tensor(self):
        l = math.layout({'a': [1, 2], 'b': (0, 1)}, channel('letter,index'))
        lt = l * wrap([1, -1])
        math.assert_close(l, lt.vector[0])
        math.assert_close([2, -2, 1, -1], math.flatten(lt.index[1]))

    def test_cast_layout(self):
        l = math.layout({'a': [1, 2], 'b': (0, 1)}, channel('letter,index'))
        lt = l * wrap([1, -1])
        math.assert_close([True, True, False, True], math.flatten(math.cast(l, bool)))
        for l in [l, lt]:
            math.assert_close(l, math.cast(l, float))

    def test_layout_to_primitive(self):
        self.assertEqual(1, int(math.layout(1.5)))
        self.assertEqual(1.5, float(math.layout(1.5)))
        self.assertEqual(1.5, complex(math.layout(1.5)))
        self.assertTrue(math.layout('a'))
        self.assertFalse(math.layout(0))
        self.assertFalse(math.layout(''))

    def test_expand_layout(self):
        layout = math.layout(['a', 'b'], spatial('alphabet'))
        exp = math.expand(layout, batch(b=10))
        self.assertEqual(batch(b=10) & spatial(alphabet=2), exp.shape)
        self.assertEqual(exp.native('b,alphabet'), [['a', 'b']] * 10)
        # self.assertEqual(exp.native('alphabet,b'), [['a'] * 10, ['b'] * 10])

    def test_layout_equality(self):
        l1 = wrap(['a', 'b'])
        l2 = wrap(['a', 'c'])
        math.assert_close(wrap([True, False]), l1 == l2)
        self.assertIsInstance(l1 == l2, math.Tensor)
        math.assert_close(wrap([False, True]), l1 != l2)
        self.assertIsInstance(l1 != l2, math.Tensor)

    def test_layout_reduce(self):
        l = math.layout([])
        self.assertFalse(l.all)
        self.assertFalse(l.any)

        l = math.layout('Hi')
        self.assertTrue(l.all)
        self.assertTrue(l.any)

        l = math.layout([0, 1, 2], channel('vector'))
        self.assertFalse(l.all)
        self.assertTrue(l.any)
        math.assert_close(1, l.mean)
        math.assert_close(math.sqrt(2/3), l.std)
        math.assert_close(0, l.min)
        math.assert_close(2, l.max)

    def test_numpy_asarray(self):
        for t in [
            tensor(.5),
            vec(x=1, y=2),
            math.random_normal(batch(b=10), spatial(x=4, y=3))
        ]:
            numpy.asarray(t)

    def test_numpy_function_interface(self):
        t = stack([1, 2, 3], channel('vector'))
        a = numpy.asarray(1.)
        for x, y in [(a, t), (t, a)]:
            math.assert_close(x + y, np.add(x, y))
            math.assert_close(x - y, np.subtract(x, y))
            math.assert_close(x * y, np.multiply(x, y))
            math.assert_close(x / y, np.divide(x, y))
            math.assert_close(x // y, np.floor_divide(x, y))
            math.assert_close(x % y, np.remainder(x, y))
            math.assert_close(x == y, np.equal(x, y))
            math.assert_close(x != y, np.not_equal(x, y))
            math.assert_close(x > y, np.greater(x, y))
            math.assert_close(x >= y, np.greater_equal(x, y))
            math.assert_close(x < y, np.less(x, y))
            math.assert_close(x <= y, np.less_equal(x, y))
            with equality_by_ref():
                self.assertFalse(x == y)
                self.assertFalse(np.equal(x, y))
                self.assertTrue(x != y)
                self.assertTrue(np.not_equal(x, y))

    def test_broadcast_vector_tuple(self):
        t = vec('something', x=0, y=1)
        i = math.linspace(0, 1, instance(particles=10))
        result = t + (0, i)
        self.assertEqual(instance(particles=10) & channel(something='x,y'), result.shape)

    def test_broadcast_matrix_tuple(self):
        i = math.linspace(0, 1, instance(particles=10))
        t = math.expand(vec(x=0, y=1), channel(c=2))
        result = t + (0, i)
        self.assertEqual(instance(particles=10) & channel(vector='x,y', c=2), result.shape)

    def test_tensor_expand_vararg(self):
        dims = [spatial(x=4, y=3), channel(vector='x,y')]
        zeros = math.zeros(*dims)
        wrapped = math.expand(0, *dims)
        tens = math.expand(tensor(0), *dims)
        math.assert_close(zeros, wrapped, tens)
        self.assertEqual(zeros.shape, wrapped.shape)
        self.assertEqual(zeros.shape, tens.shape)

    def test_auto_layout(self):
        t = wrap(['a', object()])
        self.assertEqual(channel(vector=2), t.shape)
        try:
            tensor(['a', object()])
            raise RuntimeError
        except AssertionError:
            pass

    def test_bit_shift(self):
        ints = math.range_tensor(spatial(x=4))
        math.assert_close([0, 2, 4, 6], ints << 1)
        math.assert_close(ints, (ints << 2) >> 2)
        math.assert_close([1, 2, 4, 8], 1 << ints)
        math.assert_close(1, (1 << ints) >> ints)
        np1 = np.int32(1)
        math.assert_close([0, 2, 4, 6], ints << np1)
        math.assert_close(ints, (ints << np1) >> np1)
        math.assert_close([1, 2, 4, 8], np1 << ints)
        math.assert_close(1, (np1 << ints) >> ints)
        math.assert_close([1, 0, 0, 0], np1 >> ints)

    def test_broadcast_format(self):
        t = math.range(spatial(x=5))
        scenes = -math.f-f"runs/collision_{t+1}/sim_{t:.1f}"
        for i, scene in enumerate(scenes.x):
            self.assertEqual(f"runs/collision_{i+1}/sim_{i:.1f}", scene)
        self.assertEqual('a', -math.f-'a')  # nothing to format

    def test_slice_outside(self):
        t = math.random_normal(spatial(x=3)).x[:5]
        self.assertEqual(t.x.size, 3)
