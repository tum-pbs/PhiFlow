from unittest import TestCase
from phi.flow import *


class TestSchroedinger(TestCase):

    def test_normalization_flag(self):
        q = QuantumWave(Domain([64, 64]))
        self.assertEqual(q.is_normalized, False)
        np.testing.assert_equal(q.amplitude.shape, [1, 64, 64, 1])
        q = q.copied_with(amplitude=0, is_normalized=True)
        self.assertEqual(q.is_normalized, True)
        np.testing.assert_equal(q.amplitude.shape, [1, 64, 64, 1])
        q = q.copied_with(amplitude=1)
        self.assertEqual(q.is_normalized, False)
        q = SCHROEDINGER.step(q, obstacles=[Obstacle(box[0:0, 0:0])])
        self.assertEqual(q.is_normalized, True)
        np.testing.assert_equal(q.amplitude.shape, [1, 64, 64, 1])
        np.testing.assert_almost_equal(q.amplitude[0, 0, 0, 0], 0)
        np.testing.assert_almost_equal(q.amplitude[0, 10, 10, 0], 1.0/(64-2*SCHROEDINGER.margin))