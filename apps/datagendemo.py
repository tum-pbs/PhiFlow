from phi.tf.flow import *


world.batch_size = 4


howto = """
Press 'Play' to continuously generate data until stopped.
Each step computes one frame for each scene in the batch (batch_size=%d).

Press 'Run sequence' to generate as many batches as specified in 'Sequence length'.

Each generated scene contains 'Substeps' many frames.
""" % world.batch_size


class DataGen(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, 'DataGen', howto, stride=16, base_dir='~/phi/data')
        self.smoke = world.Smoke(Domain([100, 64]), density=randn(levels=(0,0,1)))
        self.add_field('Density', lambda: self.smoke.density)
        self.add_field('Velocity', lambda: self.smoke.velocity)
        self.add_field('Domain', lambda: self.smoke.domaincache.accessible(1))
        self.add_field('Pressure', lambda: self.smoke.last_pressure)

    def step(self):
        if self.steps >= self.sequence_stride:
            self.new_scene()
            self.steps = 0
            self.smoke.density = randn(levels=(0,0,1))
            self.smoke.velocity = 0
            self.info('Starting data generation in scene %s' % self.scene)
        world.step()
        self.scene.write(self.smoke.state, frame=self.steps)


DataGen().show(display=('Density', 'Velocity'), sequence_count=4)