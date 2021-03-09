from unittest import TestCase

import phi


class TestCIInstallation(TestCase):

    def test_detect_tf_torch_jax(self):
        backends = phi.detect_backends()
        names = [b.name for b in backends]
        self.assertIn('PyTorch', names)
        self.assertIn('Jax', names)
        self.assertIn('TensorFlow', names)

    def test_verify(self):
        phi.verify()
