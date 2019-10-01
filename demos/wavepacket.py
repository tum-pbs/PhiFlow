# coding=utf-8
from phi.flow import *


scale = 4


class SchroedingerDemo(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, u'Schrödinger Demo', stride=10)
        q = self.q = world.QuantumWave(Domain([128*scale, 128*scale]))
        self.value_mass = 0.2
        self.value_frequency = 1.0
        self.value_size = 6.0
        self.action_reset()
        self.value_dt = 1.0
        glassbar = world.StepPotential(box[30*scale:50*scale, 0:1024], height=1+0j)
        topbar = world.Obstacle(box[80*scale:90*scale, 0:1024])

        self.add_field('Real', lambda: np.real(q.amplitude))
        self.add_field('Imag', lambda: np.imag(q.amplitude))
        self.add_field('Domain', lambda: geometry_mask([glassbar.field.bounds, topbar.geometry], q.grid))
        self.add_field('Zoomed', lambda: np.real(q.amplitude)[:, 0:128, 0:128, :])
        self.info('Total probability: %f' % sum(abs(self.q.amplitude)**2))

    def step(self):
        self.q.mass = self.value_mass
        world.step(dt=self.value_dt)
        self.info('Total probability: %f' % sum(abs(self.q.amplitude)**2))

    def action_reset(self):
        self.steps = 0
        self.q.amplitude = normalize_probability(wave_packet(self.q.domain, [50, 50], self.value_size,
                                                             [1*self.value_frequency, 0.6*self.value_frequency]))


SchroedingerDemo().show(figure_builder=PlotlyFigureBuilder(batches=[0], depths=[0], max_resolution=128))