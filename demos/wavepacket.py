# coding=utf-8
from phi.flow import *


scale = 4


class WavePacketDemo(App):

    def __init__(self):
        App.__init__(self, u'Schrödinger Demo', stride=10)
        q = self.q = world.add(QuantumWave(Domain([128*scale, 128*scale])))
        self.value_mass = 0.2
        self.value_frequency = 1.0
        self.value_size = 6.0
        self.action_reset()
        self.value_dt = 1.0
        glassbar = world.add(StepPotential(box[30*scale:50*scale, 0:1024], height=1+0j))
        topbar = world.add(Obstacle(box[80*scale:90*scale, 0:1024]))
        dom = GeometryMask('', glassbar.field.geometries + (topbar.geometry,)).at(q.amplitude)

        self.add_field('Real', lambda: math.real(q.amplitude))
        self.add_field('Imag', lambda: math.imag(q.amplitude))
        self.add_field('Probability', lambda: psquare(q.amplitude.data))
        self.add_field('Domain', lambda: dom)
        self.add_field('Zoomed', lambda: math.real(q.amplitude.data)[:, 0:128, 0:128, :])

    def step(self):
        self.q.mass = self.value_mass
        world.step(dt=self.value_dt)

    def action_reset(self):
        self.steps = 0
        self.q.amplitude = WavePacket([50, 50], self.value_size, [1*self.value_frequency, 0.6*self.value_frequency]) # normalize_probability(wave_packet(self.q.domain, ))


show()
