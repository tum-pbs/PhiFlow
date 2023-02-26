from typing import Tuple
from unittest import TestCase

import numpy

import phi
from phi.math.backend import ComputeDevice, convert, Backend

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

    def test_gather(self):
        for b in BACKENDS:
            t = b.zeros((4, 3, 2))
            indices = [0, 1]
            result = b.gather(t, indices, axis=0)
            self.assertEqual((2, 3, 2), b.staticshape(result))

    def test_sparse(self):
        idx = [[0, 1, 1],
               [2, 0, 2]]
        v = [3, 4, 5]
        shape = (2, 3)
        for b in BACKENDS:
            if b.supports(Backend.sparse_coo_tensor):
                with b:
                    idx_ = b.transpose(b.as_tensor(idx), [1, 0])
                    matrix = b.sparse_coo_tensor(idx_, v, shape)
                    self.assertTrue(b.is_tensor(matrix), b.name)

    def test_get_diagonal(self):
        for b in BACKENDS:
            with b:
                t = b.as_tensor([[[[1], [2]], [[0], [-1]]]])
                d = b.numpy(b.get_diagonal(t, offset=0))
                numpy.testing.assert_equal([[[1], [-1]]], d)
                d1 = b.numpy(b.get_diagonal(t, offset=1))
                numpy.testing.assert_equal([[[2]]], d1)
                d1 = b.numpy(b.get_diagonal(t, offset=-1))
                numpy.testing.assert_equal([[[0]]], d1)

    def test_solve_triangular_dense(self):
        for b in BACKENDS:
            with b:
                rhs = b.as_tensor([[1, 7, 3]])
                matrix = b.as_tensor([[[-1, 1, 0], [0, 2, 2], [0, 1, 1]]])
                x = b.numpy(b.solve_triangular_dense(matrix, rhs, lower=False, unit_diagonal=True)[0, :])
                numpy.testing.assert_almost_equal([0, 1, 3], x, err_msg=b.name)
                x = b.numpy(b.solve_triangular_dense(matrix, rhs, lower=False, unit_diagonal=False)[0, :])
                numpy.testing.assert_almost_equal([-.5, .5, 3], x, err_msg=b.name)

    def test_while_loop_direct(self):
        for b in BACKENDS:
            with b:
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
                with b:
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
