from phi.flow import *


class Simpleplume(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, 'Simpleplume', stride=5)
        smoke = world.Smoke(Domain([80, 64], SLIPPERY))
        world.Inflow(Sphere((10, 32), 5), rate=0.2)
        self.add_field('Density', lambda: smoke.density)
        self.add_field('Velocity', lambda: smoke.velocity)


app = Simpleplume().show(production=__name__ != '__main__')
