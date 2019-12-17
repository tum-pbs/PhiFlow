from phi.flow import *


world.batch_size = 4


HOW_TO = """
Press 'Play' to continuously generate data until stopped.
Each step computes one frame for each scene in the batch (batch_size=%d).

The number of frames per simulation can be adjusted in the model parameters section.

The text box next to 'Play' lets you choose how many frames you want to generate in total. It should be a multiple of the frames per simulation.
""" % world.batch_size


def random_density(shape):
    return math.maximum(0, math.randfreq(shape, power=32))


def random_velocity(shape):
    return math.randfreq(shape, power=32) * 2


class SmokeDataGen(App):

    def __init__(self):
        App.__init__(self, 'Smoke Data Generation', HOW_TO, base_dir='~/phi/data', summary='smoke')
        self.smoke = world.add(Fluid(Domain([64, 64]), density=random_density, velocity=random_velocity, batch_size=world.batch_size, buoyancy_factor=0.1), physics=IncompressibleFlow())
        self.value_frames_per_simulation = 16
        self.add_field('Density', lambda: self.smoke.density)
        self.add_field('Velocity', lambda: self.smoke.velocity)

    def step(self):
        if self.steps >= self.value_frames_per_simulation:
            self.new_scene()
            self.steps = 0
            self.smoke.density = random_density
            self.smoke.velocity = random_velocity
            self.info('Starting data generation in scene %s' % self.scene)
        world.step()
        self.scene.write(self.smoke.state, frame=self.steps)


show(SmokeDataGen(), display=('Density', 'Velocity'))
