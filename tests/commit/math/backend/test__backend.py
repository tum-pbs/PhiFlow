from unittest import TestCase

import phi
from phi.math.backend import ComputeDevice

BACKENDS = phi.detect_backends()


class TestBackends(TestCase):

    def test_list_devices(self):
        for backend in BACKENDS:
            devices = backend.list_devices()
            self.assertGreater(len(devices), 0)
            self.assertTrue(all(isinstance(d, ComputeDevice) for d in devices))
