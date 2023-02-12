from unittest import TestCase

import numpy

import phi
from phi.math.backend import ComputeDevice, convert, Backend

BACKENDS = phi.detect_backends()


class TestBackends(TestCase):

    def test_list_devices(self):
        for backend in BACKENDS:
            devices = backend.list_devices()
            self.assertGreater(len(devices), 0)
            self.assertTrue(all(isinstance(d, ComputeDevice) for d in devices))

    def test_convert(self):  # TODO this causes RuntimeError when GPU capsule is given to Jax in CPU mode
        for source_backend in BACKENDS:
            for target_backend in BACKENDS:
                data = source_backend.random_normal([4], None)  # may be deleted in conversion
                print(f"{source_backend} -> {target_backend}   {data}")
                converted = convert(data, target_backend)
                np1 = source_backend.numpy(data)
                np2 = target_backend.numpy(converted)
                numpy.testing.assert_equal(np1, np2)

    def test_allocate_on_device(self):
        for backend in BACKENDS:
            t = backend.zeros(())
            assert backend.get_device(t) == backend.get_default_device()
            t_ = backend.allocate_on_device(t, backend.get_default_device())
            assert backend.get_device(t_) == backend.get_default_device()

    def test_gather(self):
        for backend in BACKENDS:
            t = backend.zeros((4, 3, 2))
            indices = [0, 1]
            result = backend.gather(t, indices, axis=0)
            self.assertEqual((2, 3, 2), backend.staticshape(result))

    def test_sparse(self):
        idx = [[0, 1, 1],
               [2, 0, 2]]
        v = [3, 4, 5]
        shape = (2, 3)
        for backend in BACKENDS:
            if backend.supports(Backend.sparse_coo_tensor):
                with backend:
                    idx_ = backend.transpose(backend.as_tensor(idx), [1, 0])
                    matrix = backend.sparse_coo_tensor(idx_, v, shape)
                    self.assertTrue(backend.is_tensor(matrix), backend.name)

    def test_get_diagonal(self):
        for backend in BACKENDS:
            with backend:
                t = backend.as_tensor([[[[1], [2]], [[0], [-1]]]])
                d = backend.numpy(backend.get_diagonal(t, offset=0))
                numpy.testing.assert_equal([[[1], [-1]]], d)
                d1 = backend.numpy(backend.get_diagonal(t, offset=1))
                numpy.testing.assert_equal([[[2]]], d1)
                d1 = backend.numpy(backend.get_diagonal(t, offset=-1))
                numpy.testing.assert_equal([[[0]]], d1)
