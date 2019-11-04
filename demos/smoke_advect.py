from phi.physics.flow import *


class Simpleplume(App):

    def __init__(self):
        App.__init__(self, 'Smoke-Advect', stride=5)
        self.smoke = world.Smoke(Domain([80, 64], SLIPPERY))
        self.dt = 1.0

        world.Inflow(Sphere((10, 32), 5), rate=0.2)

        # create a new centered grid like the density with a simple shape to add every time step
        self.myInflow = math.zeros(self.smoke.domain.shape())
        self.myInflow[..., 5:20, 20:40, 0] = 1.

        # create a second centered grid that will be advected
        self.myField = math.zeros(self.smoke.domain.shape()) + self.myInflow
        # alternatively with batch size:
        #self.myField = initialize_field( 0., self.smoke.domain.shape(1, self.smoke._batch_size)) + self.myInflow

        self.add_field('Density', lambda: self.smoke.density)
        self.add_field('Velocity', lambda: self.smoke.velocity)
        self.add_field('New, passively advected field', lambda: self.myField)
        self.info('Init smoke advect done, output scene %s' % self.scene)

    def step(self):
        world.step(dt=self.dt)
        tmp = self.myField + self.myInflow
        self.myField = self.smoke.velocity.advect(tmp, dt=self.dt)
        self.info('Running smoke advect, step %d' % self.steps)


app = Simpleplume().show(production=__name__ != '__main__')
