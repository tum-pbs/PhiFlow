from unittest import TestCase

from phi.math import batch, get_sparsity, expand, wrap, stack, zeros, channel, spatial, ones, instance
from phi.math._sparse import SparseCoordinateTensor, CompressedSparseTensor


class TestSprase(TestCase):

    def test_sparsity(self):
        self.assertEqual(1, get_sparsity(wrap(1)))
        self.assertEqual(0.25, get_sparsity(expand(1., batch(b=4))))
        self.assertEqual(0.25, get_sparsity(stack([zeros(batch(b=4))] * 3, channel('vector'))))
        self.assertEqual(0.3, get_sparsity(SparseCoordinateTensor(ones(instance(nnz=3), channel(vector='x')), ones(instance(nnz=3)), spatial(x=10), True, False)))
        self.assertEqual(0.03, get_sparsity(CompressedSparseTensor(ones(instance(nnz=3), channel(vector='x')), ones(instance(y_pointers=4)), spatial(y=10), ones(instance(nnz=3)), spatial(x=10, y=10))))
