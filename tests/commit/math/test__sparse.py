from unittest import TestCase

import phi
from phi import math
from phi.math import batch, get_sparsity, expand, wrap, stack, zeros, channel, spatial, ones, instance, tensor, sum, pairwise_distances, vec_length, dense, assert_close
from phi.math._sparse import SparseCoordinateTensor, CompressedSparseTensor

BACKENDS = phi.detect_backends()


class TestSparse(TestCase):

    def test_sparsity(self):
        # self.assertEqual(1, get_sparsity(wrap(1)))
        # self.assertEqual(0.25, get_sparsity(expand(1., batch(b=4))))
        # self.assertEqual(0.25, get_sparsity(stack([zeros(batch(b=4))] * 3, channel('vector'))))
        # self.assertEqual(0.3, get_sparsity(SparseCoordinateTensor(ones(instance(nnz=3), channel(vector='x')), ones(instance(nnz=3)), spatial(x=10), True, False)))
        self.assertEqual(0.03, get_sparsity(CompressedSparseTensor(indices=ones(instance(nnz=3)),
                                                                   pointers=ones(instance(y_pointers=4)),
                                                                   values=ones(instance(nnz=3)),
                                                                   uncompressed_dims=spatial(x=10),
                                                                   compressed_dims=spatial(y=10))))

    def test_csr(self):
        for backend in BACKENDS:
            with backend:
                indices = tensor([0, 1, 0], instance('nnz'))
                pointers = tensor([0, 2, 3, 3], instance('pointers'))
                values = tensor([2, 3, 4], instance('nnz'))
                matrix = CompressedSparseTensor(indices, pointers, values, channel(right=3), channel(down=3))
                math.print(dense(matrix))
                assert_close((2, 3, 0), dense(matrix).down[0])
                assert_close((4, 0, 0), dense(matrix).down[1])
                assert_close((0, 0, 0), dense(matrix).down[2])
                # Multiplication
                assert_close((5, 4, 0), matrix.right * (1, 1, 1))
                # Simple arithmetic
                assert_close(matrix, (matrix + matrix * 2) / 3)

