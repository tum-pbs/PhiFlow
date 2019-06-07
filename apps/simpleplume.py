from phi.flow import *


smoke = Smoke(Domain([80, 64], SLIPPERY))
inflow(Sphere((10,32), 5), rate=0.2)


class Simpleplume(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, "Simpleplume", stride=5)
        self.state = zeros(smoke.shape())
        self.add_field("Density", lambda: self.state.density)
        self.add_field("Velocity", lambda: self.state.velocity)

    def step(self):
        self.state = smoke.step(self.state)


app = Simpleplume().show(production=__name__ != "__main__")
