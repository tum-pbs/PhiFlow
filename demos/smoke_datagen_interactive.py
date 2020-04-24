from phi.flow import *


world.batch_size = 4


HOW_TO = """
Press 'Play' to continuously generate data until stopped.
Each step computes one frame for each scene in the batch (batch_size=%d).

The number of frames per simulation can be adjusted in the model parameters section.

The text box next to 'Play' lets you choose how many frames you want to generate in total. It should be a multiple of the frames per simulation.
""" % world.batch_size


class SmokeDataGen(App):

    def __init__(self):
        App.__init__(self, 'Smoke Data Generation', HOW_TO, base_dir='~/phi/data', summary='smoke')
        self.smoke = world.add(Fluid(Domain([64, 64]), density=math.maximum(0, Noise() * 0.3), velocity=Noise() * 0.5, batch_size=world.batch_size, buoyancy_factor=0.1), physics=IncompressibleFlow())
        self.value_frames_per_simulation = 16

    def step(self):
        if self.steps >= self.value_frames_per_simulation:
            self.new_scene()
            self.steps = 0
            self.smoke.density = math.maximum(0, Noise() * 0.3)
            self.smoke.velocity = Noise() * 0.5
            self.info('Starting data generation in scene %s' % self.scene)
        world.step()
        self.scene.write(self.smoke.state, frame=self.steps)


show()
