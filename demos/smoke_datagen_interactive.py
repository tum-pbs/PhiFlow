from phi.tf.flow import *


world.batch_size = 4


howto = """
Press 'Play' to continuously generate data until stopped.
Each step computes one frame for each scene in the batch (batch_size=%d).

Press 'Run sequence' to generate as many batches as specified in 'Sequence length'.

Each generated scene contains 'Substeps' many frames.
""" % world.batch_size


random_density = lambda shape: math.maximum(0, math.randfreq(shape, power=32))
random_velocity = lambda shape: math.randfreq(shape, power=32) * 2


class SmokeDataGen(App):

    def __init__(self):
        App.__init__(self, 'SmokeDataGen', howto, stride=16, base_dir='~/phi/data', summary='smoke')
        self.smoke = world.add(Smoke(Domain([64, 64]), density=random_density, velocity=random_velocity))
        self.add_field('Density', lambda: self.smoke.density)
        self.add_field('Velocity', lambda: self.smoke.velocity)
        self.add_field('Domain', lambda: self.smoke.domaincache.accessible(1))
        self.add_field('Pressure', lambda: self.smoke.last_pressure)

    def step(self):
        if self.steps >= self.sequence_stride:
            self.new_scene()
            self.steps = 0
            self.smoke.density = random_density
            self.smoke.velocity = random_velocity
            self.info('Starting data generation in scene %s' % self.scene)
        world.step()
        self.scene.write(self.smoke.state, frame=self.steps)


show(SmokeDataGen(), display=('Density', 'Velocity'), sequence_count=4)
