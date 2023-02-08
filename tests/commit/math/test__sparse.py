from unittest import TestCase

import phi
from phi import math
from phi.math import batch, get_sparsity, expand, wrap, stack, zeros, channel, spatial, ones, instance, tensor, sum, pairwise_distances, vec_length, dense, assert_close, non_dual
from phi.math._sparse import SparseCoordinateTensor, CompressedSparseMatrix

BACKENDS = phi.detect_backends()


class TestSparse(TestCase):

    def test_sparsity(self):
        self.assertEqual(1, get_sparsity(wrap(1)))
        self.assertEqual(0.25, get_sparsity(expand(1., batch(b=4))))
        self.assertEqual(0.25, get_sparsity(stack([zeros(batch(b=4))] * 3, channel('vector'))))
        self.assertEqual(0.3, get_sparsity(SparseCoordinateTensor(ones(instance(nnz=3), channel(vector='x'), dtype=int), ones(instance(nnz=3)), spatial(x=10), True, False)))
        self.assertEqual(0.03, get_sparsity(CompressedSparseMatrix(indices=ones(instance(nnz=3), dtype=int),
                                                                   pointers=ones(instance(y_pointers=11), dtype=int),
                                                                   values=ones(instance(nnz=3)),
                                                                   uncompressed_dims=spatial(x=10),
                                                                   compressed_dims=spatial(y=10))))

    def test_csr(self):
        for backend in BACKENDS:
            with backend:
                indices = tensor([0, 1, 0], instance('nnz'))
                pointers = tensor([0, 2, 3, 3], instance('pointers'))
                values = tensor([2, 3, 4], instance('nnz'))
                matrix = CompressedSparseMatrix(indices, pointers, values, channel(right=3), channel(down=3))
                math.print(dense(matrix))
                assert_close((2, 3, 0), dense(matrix).down[0])
                assert_close((4, 0, 0), dense(matrix).down[1])
                assert_close((0, 0, 0), dense(matrix).down[2])
                # Multiplication
                assert_close((5, 4, 0), matrix.right * (1, 1, 1))
                # Simple arithmetic
                assert_close(matrix, (matrix + matrix * 2) / 3)

    def test_csr_slice_concat(self):
        pos = tensor([(0, 0), (0, 1), (0, 2)], instance('particles'), channel(vector='x,y'))
        dx = math.pairwise_distances(pos, max_distance=1.5, format='csr')
        self.assertEqual(0, dx.sum)
        dist = math.vec_length(dx, eps=1e-6)
        self.assertEqual(instance(particles=3, others=3), dist.shape)
        self.assertGreater(dist.sum, 0)
        # Slice channel
        dx_y = dx['y']
        self.assertEqual(instance(particles=3, others=3), dx_y.shape)
        # Slice / concat compressed
        concat_particles = math.concat([dx.particles[:1], dx.particles[1:]], 'particles')
        math.assert_close(dx, concat_particles)
        # Slice / concat uncompressed
        concat_others = math.concat([dx.others[:1], dx.others[1:]], 'others')
        math.assert_close(dx, concat_others)

    def test_coo(self):
        def f(x):
            return math.laplace(x)

        for backend in BACKENDS:
            with backend:
                x = math.ones(spatial(x=5))
                coo, bias = math.matrix_from_function(f, x, auto_compress=False)
                csr = coo.compress(non_dual)
                math.assert_close(f(x), coo @ x, csr @ x)