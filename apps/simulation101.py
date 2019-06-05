from phi.flow import *
from phi.model import FieldSequenceModel


smoke = Smoke(Domain([64, 64], SLIPPERY))
inflow(Sphere((10,32), 5), rate=0.2)


class Simulation101(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, "PhiFlow - Simulation 101", "Smoke simulation with NumPy", stride=5)
        self.state = smoke.empty()
        self.add_field("Density", lambda: self.state.density)
        self.add_field("Velocity", lambda: self.state.velocity)

    def step(self):
        self.state = smoke.step(self.state)


app = Simulation101().show(production=__name__ != "__main__")
