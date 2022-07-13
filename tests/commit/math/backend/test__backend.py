from unittest import TestCase

import numpy

import phi
from phi.math.backend import ComputeDevice, convert


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

