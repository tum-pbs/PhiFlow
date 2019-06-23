from phi.flow import *


class Simpleplume(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, "Simpleplume", stride=5)
        self.smoke = world.Smoke(Domain([80, 64], SLIPPERY))
        world.Inflow(Sphere((10, 32), 5), rate=0.2)
        self.add_field("Density", lambda: self.smoke.density)
        self.add_field("Velocity", lambda: self.smoke.velocity)

    def step(self):
        world.step()


app = Simpleplume().show(production=__name__ != "__main__")
