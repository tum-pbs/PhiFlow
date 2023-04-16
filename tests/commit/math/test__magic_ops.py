from typing import Tuple, Union
from unittest import TestCase

import dataclasses

from phi import math

from phi.math import batch, unstack, Shape, merge_shapes, stack, concat, expand, spatial, shape, instance, rename_dims, \
    pack_dims, random_normal, flatten, unpack_dim, EMPTY_SHAPE, Tensor, Dict, channel, linspace, zeros, meshgrid, assert_close, wrap, vec
from phi.math._magic_ops import bool_to_int
from phi.math.magic import BoundDim, Shaped, Sliceable, Shapable, PhiTreeNode, slicing_dict


class Stackable:

    def __init__(self, shape: Shape):
        self.shape = shape

    def __getitem__(self, item: dict):
        return Stackable(self.shape.after_gather(slicing_dict(self, item)))

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **kwargs) -> 'Stackable':
        return Stackable(merge_shapes(dim, *[v.shape for v in values]))


class ConcatExpandable:

    def __init__(self, shape: Shape):
        self.shape = shape

    def __getitem__(self, item: dict):
        return ConcatExpandable(self.shape.after_gather(slicing_dict(self, item)))

    @staticmethod
    def __concat__(values: tuple, dim: str, **kwargs) -> 'ConcatExpandable':
        try:
            new_size = sum([v.shape.get_item_names(dim) for v in values], ())
        except:
            new_size = sum([v.shape.get_size(dim) for v in values])
        return ConcatExpandable(values[0].shape.with_dim_size(dim, new_size))

    def __expand__(self, dims: Shape, **kwargs) -> 'ConcatExpandable':
        return ConcatExpandable(merge_shapes(dims, self.shape))


class ValuedPhiTreeNode:

    def __init__(self, shape: Shape):
        self.tensor = zeros(shape)

    def __value_attrs__(self) -> Tuple[str, ...]:
        return 'tensor',

    def __getitem__(self, item: dict):
        return ValuedPhiTreeNode(self.tensor[item].shape)


@dataclasses.dataclass
class MyPoint:
    x: Tensor or float
    y: Union[Tensor, float]
    is_normalized: bool = dataclasses.field(compare=False)

    def __getitem__(self, item):
        return MyPoint(self.x[item], self.y[item], is_normalized=self.is_normalized)


TEST_CLASSES = [
    Stackable,
    ConcatExpandable,
    random_normal,
    ValuedPhiTreeNode,
    lambda shape: MyPoint(zeros(shape), zeros(shape), is_normalized=False),
]


class TestMagicOps(TestCase):

    def test_unstack_not_implemented(self):

        class TestClass:

            def __init__(self, custom_unstack: bool, shape=batch(dim1=2, dim2=3)):
                self.custom_unstack = custom_unstack
                self.shape = shape

            def __getitem__(self, item: dict):
                return TestClass(self.custom_unstack, self.shape.after_gather(item))

            def __unstack__(self, dims: Tuple[str, ...]) -> Tuple['Sliceable', ...]:
                if not self.custom_unstack:
                    return NotImplemented
                else:
                    return unstack(TestClass(False, self.shape), dims)

        t = unstack(TestClass(True), 'dim1')
        self.assertIsInstance(t, tuple)
        self.assertEqual(2, len(t))
        t = unstack(TestClass(True), 'dim1,dim2')
        self.assertIsInstance(t, tuple)
        self.assertEqual(6, len(t))

    def test_subclasscheck(self):
        self.assertTrue(issubclass(Stackable, Shaped))
        self.assertTrue(issubclass(Stackable, Sliceable))
        self.assertTrue(issubclass(Stackable, Shapable))
        self.assertFalse(issubclass(object, Shapable))
        self.assertFalse(issubclass(object, Sliceable))
        self.assertTrue(issubclass(ConcatExpandable, Shaped))
        self.assertTrue(issubclass(ConcatExpandable, Sliceable))
        self.assertTrue(issubclass(ConcatExpandable, Shapable))
        self.assertTrue(issubclass(ValuedPhiTreeNode, Shaped))
        self.assertTrue(issubclass(ValuedPhiTreeNode, Sliceable))
        self.assertTrue(issubclass(ValuedPhiTreeNode, Sliceable))
        self.assertTrue(issubclass(MyPoint, Shaped))
        self.assertTrue(issubclass(MyPoint, Sliceable))
        self.assertTrue(issubclass(MyPoint, Sliceable))

    def test_instancecheck(self):
        for test_class in TEST_CLASSES:
            a = test_class(spatial(x=5) & batch(b=2))
            self.assertIsInstance(a, Shaped)
            self.assertIsInstance(a, Sliceable)
            self.assertIsInstance(a, Shapable)
        self.assertNotIsInstance('test', Shapable)
        self.assertNotIsInstance('test', Sliceable)
        class S:
            def __shape__(self):
                return batch()
        self.assertIsInstance(S(), Shaped)
        self.assertNotIsInstance(S(), Shapable)

    def test_shape(self):
        for test_class in TEST_CLASSES:
            a = test_class(spatial(x=5) & batch(b=2))
            self.assertEqual(spatial(x=5) & batch(b=2), shape(a))

    def test_slice(self):
        for test_class in TEST_CLASSES:
            a = test_class(spatial(x=5) & channel(vector='x,y,z') & batch(b=2))
            self.assertEqual(spatial(x=5) & batch(b=2), shape(a['x']))
            self.assertEqual(spatial(x=5) & batch(b=2) & channel(vector='y,z'), shape(a['y,z']))

    def test_unstack(self):
        for test_class in TEST_CLASSES:
            a = test_class(spatial(x=5) & batch(b=2))
            u = unstack(a, 'b')
            self.assertEqual(2, len(u))
            self.assertEqual(spatial(x=5), shape(u[0]))
            self.assertEqual(10, len(unstack(a, 'x,b')))
            self.assertEqual(10, len(unstack(a, 'b,x')))

    def test_stack(self):
        for test_class in TEST_CLASSES:
            a = test_class(spatial(x=5))
            self.assertEqual(spatial(x=5) & batch(b=2), shape(stack([a, a], batch('b'))))
            self.assertEqual(spatial(x=5) & batch(b='a1,a2'), shape(stack({'a1': a, 'a2': a}, batch('b'))))

    def test_stack_expand(self):
        v = stack([0, linspace(0, 1, instance(points=10))], channel(vector='x,y'), expand_values=True)
        self.assertEqual(set(instance(points=10) & channel(vector='x,y')), set(shape(v)))

    def test_multi_dim_stack(self):
        for test_class in TEST_CLASSES:
            a = test_class(spatial(x=5))
            self.assertEqual(spatial(x=5) & batch(a=3, b=2), shape(stack([a]*6, batch(a=3, b=2))))

    def test_concat(self):
        for test_class in TEST_CLASSES:
            a = test_class(spatial(x=5) & batch(b=2))
            self.assertEqual(spatial(x=5) & batch(b=4), shape(concat([a, a], batch('b'))))

    def test_expand(self):
        for test_class in TEST_CLASSES:
            a = test_class(spatial(x=5))
            self.assertEqual(spatial(x=5) & batch(b=2), shape(expand(a, batch(b=2))))

    def test_rename_dims(self):
        for test_class in TEST_CLASSES:
            a = test_class(spatial(x=5) & batch(b=2))
            self.assertEqual(instance(points=5) & batch(b=2), shape(rename_dims(a, 'x', instance('points'))))

    def test_retype_dims(self):
        t = math.zeros(spatial(x=4, y=3), channel(vector='x,y'))
        self.assertEqual(instance(x=4, y=3) & channel(vector='x,y'), rename_dims(t, spatial, instance).shape)
        self.assertEqual(set(spatial(x=4, y=3) & batch(vector='x,y')), set(rename_dims(t, channel, batch).shape))

    def test_pack_dims(self):
        for test_class in TEST_CLASSES:
            a = test_class(spatial(x=5) & batch(b=2))
            self.assertEqual(instance(points=5) & batch(b=2), shape(pack_dims(a, 'x', instance('points'))))  # Rename / type
            self.assertEqual(instance(points=10), shape(pack_dims(a, 'x,b', instance('points'))))  # Pack
            self.assertEqual(set(spatial(x=5) & batch(b=2) & instance(points=3)), set(shape(pack_dims(a, '', instance(points=3)))))  # Un-squeeze

    def test_unpack_dim(self):
        for test_class in TEST_CLASSES:
            a = test_class(instance(points=10))
            self.assertEqual(batch(b=10), shape(unpack_dim(a, 'points', batch('b'))))  # Rename / type
            self.assertEqual(spatial(x=5) & batch(b=2), shape(unpack_dim(a, 'points', spatial(x=5) & batch(b=2))))  # Unpack
            a = test_class(instance(points=10) & batch(b=1))
            self.assertEqual(instance(points=10), shape(unpack_dim(a, 'b', EMPTY_SHAPE)))  # Squeeze

    def test_flatten(self):
        for test_class in TEST_CLASSES:
            a = test_class(spatial(x=5) & batch(b=2))
            self.assertEqual(instance(points=10), shape(flatten(a, instance('points'), flatten_batch=True)))
            self.assertEqual(batch(b=2) & instance(points=5), shape(flatten(a, instance('points'), flatten_batch=False)))

    def test_bound_dim(self):
        for test_class in TEST_CLASSES:
            a = test_class(spatial(x=5) & batch(b=2))
            x = BoundDim(a, 'x')
            self.assertEqual(spatial(y=5) & batch(b=2), shape(x.rename('y')))
            self.assertEqual(instance(x=5) & batch(b=2), shape(x.retype(instance)))
            self.assertEqual(instance(y=5) & batch(b=2), shape(x.replace(instance('y'))))
            self.assertEqual(instance(y=5) & batch(b=2), shape(x.unpack(instance('y'))))

    def test_phi_tree_subclasscheck(self):
        self.assertTrue(issubclass(Tensor, PhiTreeNode))
        self.assertTrue(issubclass(tuple, PhiTreeNode))
        self.assertTrue(issubclass(list, PhiTreeNode))
        self.assertTrue(issubclass(dict, PhiTreeNode))
        self.assertTrue(issubclass(Dict, PhiTreeNode))

    def test_bound_dims(self):
        v = meshgrid(x=4, y=3)
        assert_close(1, v.x.y.vector[1, 0, 0])
        assert ('x', 'y') == instance(v.x.y.as_instance()).names
        assert instance(points=12) & channel(vector='x,y') == v.x.y.pack(instance('points')).shape
        assert 12 == v.x.y.volume
        assert 24 == v.x.y.vector.volume
        assert 24 == len(v.x.y.vector.unstack())
        assert 24 == len(tuple(v.x.y.vector))
        try:
            v.x.y.size
            self.fail()
        except SyntaxError:
            pass

    def test_bool_to_int(self):
        a = [wrap(True), wrap(1.), {'a': wrap(False)}]
        self.assertEqual([1, 1., {'a': 0}], bool_to_int(a))

    def test_slice_tree(self):
        t1, t2 = vec(x=0, y=1), vec(x=2, y=3)
        self.assertEqual([1, 3], math.slice([t1, t2], {'vector': 'y'}))

    def test_unstack_tree(self):
        t1, t2 = vec(x=0, y=1), vec(x=2, y=3)
        self.assertEqual(([0, 2], [1, 3]), unstack([t1, t2], 'vector'))
