from phi.flow import *


class Simpleplume(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, stride=5)
        world.Smoke(Domain([80, 64], boundaries=SLIPPERY))
        world.Inflow(Sphere(center=(10, 32), radius=5), rate=0.2)


show()