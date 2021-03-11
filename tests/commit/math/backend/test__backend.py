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
                data = source_backend.random_normal([4])  # may be deleted in conversion
                print(f"{source_backend} -> {target_backend}   {data}")
                converted = convert(data, target_backend)
                np1 = source_backend.numpy(data)
                np2 = target_backend.numpy(converted)
                numpy.testing.assert_equal(np1, np2)
