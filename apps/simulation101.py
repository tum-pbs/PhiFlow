from phi.flow import *
from phi.model import FieldSequenceModel


smoke = Smoke(Domain([80, 64], SLIPPERY))


class Simulation101(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, "PhiFlow - Simulation 101", "Smoke simulation with NumPy", stride=10)
        self.state = smoke.empty()
        inflow(box[8, 12:64-12])
        self.add_field("Density", lambda: self.state.density)
        self.add_field("Velocity", lambda: self.state.velocity)
        self.step()

    def step(self):
        self.state = smoke.step(self.state)


app = Simulation101().show(production=__name__ != "__main__")
