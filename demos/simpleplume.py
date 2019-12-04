from phi.flow import *


class Simpleplume(App):

    def __init__(self):
        App.__init__(self, stride=5)
        world.add(Fluid(Domain([80, 64], boundaries=CLOSED), buoyancy_factor=0.1))
        world.add(Inflow(Sphere(center=(10, 32), radius=5), rate=0.2))


show()
