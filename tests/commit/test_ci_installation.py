from unittest import TestCase

import phi


class TestCIInstallation(TestCase):

    def test_detect_torch_jax(self):
        backends = phi.detect_backends()
        names = [b.name for b in backends]
        self.assertIn('torch', names)
        self.assertIn('jax', names)

    def test_verify(self):
        phi.verify()
