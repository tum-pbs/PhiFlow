from phi.flow import *
from phi.model import FieldSequenceModel


class Simulation101(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, "PhiFlow - Simulation 101", "Smoke simulation with NumPy", stride=10)
        self.smoke = Smoke(Domain([8, 6], SLIPPERY))
        self.add_field("Density", lambda: self.smoke.density)
        self.add_field("Velocity", lambda: self.smoke.velocity)
        self.smoke += 2 * box[2, 1:3]
        self.smoke = self.smoke.buoyancy(0.1)
        self.smoke = self.smoke.advect()
        # self.smoke.friction()
        self.smoke = self.smoke.divergence_free()

    def step(self):
        self.smoke = self.smoke.divergence_free()
        # self.smoke = (self.smoke.advect(conserve_density=True) + 2 * box[8:9, 22:64-22]).buoancy(0.1).divergence_free()


app = Simulation101().show(production=__name__ != "__main__")
