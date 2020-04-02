# coding=utf-8
from phi.flow import *


SCALE = 4


class WavePacketDemo(App):

    def __init__(self):
        App.__init__(self, u'Schrödinger Demo', framerate=10)
        wave = self.wave = world.add(QuantumWave(Domain([128 * SCALE, 128 * SCALE])), physics=Schroedinger())
        self.value_mass = 0.2
        self.value_frequency = 1.0
        self.value_size = 6.0
        self.action_reset()
        self.value_dt = 1.0
        glassbar = world.add(StepPotential(box[30*SCALE:50*SCALE, 0:1024], height=1+0j))
        topbar = world.add(Obstacle(box[80*SCALE:90*SCALE, 0:1024]))
        dom = GeometryMask(glassbar.field.geometries).at(wave.amplitude) * 0.5 + GeometryMask([topbar.geometry]).at(wave.amplitude)

        self.add_field('Real', lambda: math.real(wave.amplitude))
        self.add_field('Imag', lambda: math.imag(wave.amplitude))
        self.add_field('Probability', lambda: psquare(wave.amplitude.data))
        self.add_field('Domain', lambda: dom)
        self.add_field('Zoomed', lambda: math.real(wave.amplitude.data)[:, 0:128, 0:128, :])

    def step(self):
        self.wave.mass = self.value_mass
        world.step(dt=self.value_dt)

    def action_reset(self):
        self.steps = 0
        self.wave.amplitude = WavePacket(center=[50, 50], size=self.value_size, wave_vector=[1 * self.value_frequency, 0.6 * self.value_frequency])


show(WavePacketDemo)
