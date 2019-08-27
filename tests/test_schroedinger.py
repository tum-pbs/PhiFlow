from unittest import TestCase
from phi.flow import *


class TestSchroedinger(TestCase):

    def test_normalization_flag(self):
        q = ProbabilityAmplitude(Domain([64, 64]), real=1)
        self.assertEqual(q.is_normalized, False)
        np.testing.assert_equal(q.real.shape, [1, 64, 64, 1])
        q = q.copied_with(real=0, is_normalized=True)
        self.assertEqual(q.is_normalized, True)
        np.testing.assert_equal(q.real.shape, [1, 64, 64, 1])
        q = q.copied_with(imag=1)
        self.assertEqual(q.is_normalized, False)
        q = normalize_probability(q)
        self.assertEqual(q.is_normalized, True)
        np.testing.assert_equal(q.real.shape, [1, 64, 64, 1])
        np.testing.assert_almost_equal(q.imag, 1.0/64)