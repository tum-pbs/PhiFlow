from unittest import TestCase

from phi.flow import *


class TestSchroedinger(TestCase):

    def test_simple_step(self):
        q = QuantumWave(Domain([4, 5]))
        q = q.copied_with(amplitude=WavePacket([2, 2], 1.0, [0.5, 0]))
        q = SCHROEDINGER.step(q, 1.0)
        np.testing.assert_equal(q.amplitude.data.shape, [1, 4, 5, 1])

    def test_complex_step(self):
        q = QuantumWave(Domain([4, 4]))
        q = q.copied_with(amplitude=WavePacket([2, 2], 1.0, [0.5, 0]))
        pot = StepPotential(box[0:1, 0:1], 1.0)
        SCHROEDINGER.step(q, 1.0, potentials=[pot], obstacles=[Obstacle(box[3:4, 0:1])])
        np.testing.assert_equal(q.amplitude.data.shape, [1, 4, 4, 1])
