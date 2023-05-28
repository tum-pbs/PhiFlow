from typing import Tuple
from unittest import TestCase

import numpy

import phi
from phi.math import DType
from phi.math.backend import ComputeDevice, convert, Backend, NUMPY

BACKENDS: Tuple[Backend] = phi.detect_backends()


class TestBackends(TestCase):

    def test_list_devices(self):
        for b in BACKENDS:
            devices = b.list_devices()
            self.assertGreater(len(devices), 0)
            self.assertTrue(all(isinstance(d, ComputeDevice) for d in devices))

    def test_convert(self):  # TODO this causes RuntimeError when GPU capsule is given to Jax in CPU mode
        for b_src in BACKENDS:
            for b_target in BACKENDS:
                data = b_src.random_normal([4], None)  # may be deleted in conversion
                print(f"{b_src} -> {b_target}   {data}")
                converted = convert(data, b_target)
                np1 = b_src.numpy(data)
                np2 = b_target.numpy(converted)
                numpy.testing.assert_equal(np1, np2)

    def test_allocate_on_device(self):
        for b in BACKENDS:
            t = b.zeros(())
            assert b.get_device(t) == b.get_default_device()
            t_ = b.allocate_on_device(t, b.get_default_device())
            assert b.get_device(t_) == b.get_default_device()

    def test_unravel_index(self):
        for b in BACKENDS:
            flat = b.as_tensor([0, 2, 3])
            indices = b.numpy(b.unravel_index(flat, (1, 2, 3)))
            numpy.testing.assert_equal([(0, 0, 0), (0, 0, 2), (0, 1, 0)], indices, err_msg=b.name)

    def test_ravel_multi_index(self):
        for b in BACKENDS:
            # --- All inside ---
            indices = b.as_tensor([(0, 0, 0), (0, 0, 2), (0, 1, 0)])
            flat = b.ravel_multi_index(indices, (1, 2, 3), mode='undefined')
            numpy.testing.assert_equal([0, 2, 3], b.numpy(flat), err_msg=b.name)
            # --- Default ---
            indices = b.as_tensor([(0, 0, 0), (0, 0, -1), (0, 0, 3), (0, 1, 0)])
            flat = b.ravel_multi_index(indices, (1, 2, 3), mode=-1)
            numpy.testing.assert_equal([0, -1, -1, 3], b.numpy(flat), err_msg=b.name)
            # --- Periodic ---
            indices = b.as_tensor([(0, 0, 0), (0, 0, -1), (0, 0, 3), (0, 1, 0)])
            flat = b.ravel_multi_index(indices, (1, 2, 3), mode='periodic')
            numpy.testing.assert_equal([0, 2, 0, 3], b.numpy(flat), err_msg=b.name)
            # --- Clamp ---
            indices = b.as_tensor([(0, 0, 0), (0, 0, -1), (0, 0, 3), (0, 1, 0)])
            flat = b.ravel_multi_index(indices, (1, 2, 3), mode='clamp')
            numpy.testing.assert_equal([0, 0, 2, 3], b.numpy(flat), err_msg=b.name)

    def test_gather(self):
        for b in BACKENDS:
            t = b.zeros((4, 3, 2))
            indices = [0, 1]
            result = b.gather(t, indices, axis=0)
            self.assertEqual((2, 3, 2), b.staticshape(result))

    def test_argsort(self):
        for b in BACKENDS:
            x = b.as_tensor([1, 0, 2, 1, -5])
            perm = b.argsort(x)
            x_sorted = b.numpy(b.gather_1d(x, perm))
            numpy.testing.assert_equal([-5, 0, 1, 1, 2], x_sorted)

    def test_searchsorted(self):
        for b in BACKENDS:
            seq = b.as_tensor([1, 2, 2, 5, 5, 5])
            val = b.as_tensor([0, 6, 1, 2, 2, 3, 5])
            left = b.numpy(b.searchsorted(seq, val, side='left'))
            numpy.testing.assert_equal([0, 6, 0, 1, 1, 3, 3], left)
            right = b.numpy(b.searchsorted(seq, val, side='right'))
            numpy.testing.assert_equal([0, 6, 1, 3, 3, 3, 6], right)

    def test_sparse(self):
        idx = [[0, 1, 1],
               [2, 0, 2]]
        v = [3, 4, 5]
        shape = (2, 3)
        for b in BACKENDS:
            if b.supports(Backend.sparse_coo_tensor):
                idx_ = b.transpose(b.as_tensor(idx), [1, 0])
                matrix = b.sparse_coo_tensor(idx_, v, shape)
                self.assertTrue(b.is_tensor(matrix), b.name)
                self.assertTrue(b.is_sparse(matrix), b.name)
                self.assertFalse(b.is_sparse(b.random_normal((2, 2), DType(float, 32))), b.name)
                np_matrix = b.numpy(matrix)
                self.assertTrue(NUMPY.is_sparse(np_matrix))
                self.assertEqual(np_matrix.shape, b.staticshape(matrix))

    def test_get_diagonal(self):
        for b in BACKENDS:
            t = b.as_tensor([[[[1], [2]], [[0], [-1]]]])
            d = b.numpy(b.get_diagonal(t, offset=0))
            numpy.testing.assert_equal([[[1], [-1]]], d)
            d1 = b.numpy(b.get_diagonal(t, offset=1))
            numpy.testing.assert_equal([[[2]]], d1)
            d1 = b.numpy(b.get_diagonal(t, offset=-1))
            numpy.testing.assert_equal([[[0]]], d1)

    def test_solve_triangular_dense(self):
        for b in BACKENDS:
            rhs = b.as_tensor([[1, 7, 3]])
            matrix = b.as_tensor([[[-1, 1, 0], [0, 2, 2], [0, 1, 1]]])
            x = b.numpy(b.solve_triangular_dense(matrix, rhs, lower=False, unit_diagonal=True)[0, :])
            numpy.testing.assert_almost_equal([0, 1, 3], x, err_msg=b.name)
            x = b.numpy(b.solve_triangular_dense(matrix, rhs, lower=False, unit_diagonal=False)[0, :])
            numpy.testing.assert_almost_equal([-.5, .5, 3], x, err_msg=b.name)

    def test_while_loop_direct(self):
        for b in BACKENDS:
            # --- while loop with max_iter ---
            count, = b.while_loop(lambda i: (i - 1,), (b.as_tensor(10),), max_iter=3)
            numpy.testing.assert_almost_equal(7, b.numpy(count), err_msg=b.name)
            # --- while loop with multiple max_iter ---
            count, = b.while_loop(lambda i: (i - 1,), (b.as_tensor(10),), max_iter=[0, 3, 6])
            numpy.testing.assert_almost_equal([10, 7, 4], b.numpy(count), err_msg=b.name)
            # --- while loop with fill ---
            count, = b.while_loop(lambda i: (i - 1,), (b.as_tensor(2),), max_iter=(1, 5))
            self.assertEqual((2,), b.staticshape(count))
            numpy.testing.assert_almost_equal([1, 1], b.numpy(count), err_msg=b.name)

    def test_while_loop_jit(self):
        for b in BACKENDS:
            if b.supports(Backend.jit_compile):
                # --- while loop with max_iter ---
                def max_iter_int(start):
                    print("max_iter_int")
                    return b.while_loop(lambda i: (i - 1,), (start,), max_iter=3)
                count, = b.jit_compile(max_iter_int)(b.as_tensor(10))
                numpy.testing.assert_almost_equal(7, b.numpy(count), err_msg=b.name)
                # --- while loop with multiple max_iter ---
                def max_iter_sequence(start):
                    print("max_iter_sequence")
                    return b.while_loop(lambda i: (i - 1,), (start,), max_iter=[0, 3, 6])
                count, = b.jit_compile(max_iter_sequence)(b.as_tensor(10))
                numpy.testing.assert_almost_equal([10, 7, 4], b.numpy(count), err_msg=b.name)
                # --- while loop with fill ---
                def max_iter_none(start):
                    print("max_iter_none")
                    return b.while_loop(lambda i: (i - 1,), (start,), max_iter=(1, 5))
                count, = b.jit_compile(max_iter_none)(b.as_tensor(2))
                numpy.testing.assert_almost_equal(1, b.numpy(count)[0], err_msg=b.name)

    def test_bincount(self):
        for b in BACKENDS:
            data = b.as_tensor([0, 2, 1, 1, 2, 1])
            result = b.numpy(b.bincount(data, None, bins=3))
            numpy.testing.assert_equal([1, 3, 2], result)
            result = b.numpy(b.bincount(data, None, bins=5))
            numpy.testing.assert_equal([1, 3, 2, 0, 0], result)

    def test_vectorized_call(self):
        for b in BACKENDS:
            def gather1d(val, idx):
                return b.gather_1d(val, idx)
            values = b.as_tensor([(0, 1, 2, 3), (1, 2, 3, 4)])
            indices = b.as_tensor([(-1, 0)])
            result = b.vectorized_call(gather1d, values, indices)
            numpy.testing.assert_equal([(3, 0), (4, 1)], b.numpy(result))
            result = b.vectorized_call(gather1d, values, indices, output_dtypes=b.dtype(values))
            numpy.testing.assert_equal([(3, 0), (4, 1)], b.numpy(result))

    def test_linspace_without_last(self):
        for b in BACKENDS:
            result = b.linspace_without_last(-1, 1, 4)
            numpy.testing.assert_equal([-1, -.5, 0, .5], b.numpy(result))
