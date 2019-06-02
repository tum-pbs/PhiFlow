from phi.flow import *
from phi.model import FieldSequenceModel
from phi.geom import *


class SimpleSmoke(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, "PhiFlow - Simulation 101", "Smoke simulation with NumPy", stride=10)
        self.smoke = Fluid(Domain([64, 64], SLIPPERY), conserve_density=True)
        self.add_field("Density", lambda: self.smoke)
        self.add_field("Velocity", lambda: self.smoke.velocity)

    def step(self):
        self.smoke = (self.smoke.advect() + 2 * box[8:9, 22:64-22]).buoancy(0.1).divergence_free()



class ColoredLiquid(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, "PhiFlow - Simulation 101", "Smoke simulation with NumPy", stride=10)
        self.smoke = Fluid(Domain([64, 64], SLIPPERY), conserve_density=True)
        self.smoke = self.smoke.with_density(box[8:9, 22:64-22])
        self.smoke = self.smoke.with_field("color", self.smoke.location())  # initialize RGB with XYZ
        self.add_field("Density", lambda: self.smoke)
        self.add_field("Velocity", lambda: self.smoke.velocity)
        self.add_field("Color", lambda: self.smoke["color"])

    def step(self):
        self.smoke = self.smoke.advect().gravity(-9.81).divergence_free()


class FreeLiquid(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, "PhiFlow - Simulation 101", "Smoke simulation with NumPy", stride=10)
        self.smoke = Fluid(Open2D, conserve_density=True)
        self.add_field("Density", lambda: self.smoke)
        self.add_field("Velocity", lambda: self.smoke.velocity)

    def step(self):
        self.smoke = (self.smoke.advect() + 2 * box[8:9, 22:64-22]).buoancy(0.1).divergence_free()