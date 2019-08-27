# coding=utf-8
from phi.flow import *


dt = 0.2


class SchroedingerDemo(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, u'Schrödinger Demo', stride=2)
        q = world.ProbabilityAmplitude(Domain([128, 128]))
        q.real, q.imag = wave_packet(q.domain, [30, 40], 6, [0.5, 0.3])
        q.imag = world.step(q, dt=dt/2).imag
        # world.QuantumBarrier(box[60:80, 0:128], 1)

        self.add_field('Real', lambda: q.real)
        self.add_field('Imag', lambda: q.imag)

    def step(self):
        world.step(dt=dt)


SchroedingerDemo().show(framerate=5)