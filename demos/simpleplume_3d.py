from phi.flow import *


class Simpleplume(App):

    def __init__(self):
        App.__init__(self, framerate=10)
        world.add(Fluid(Domain([40, 32, 32], boundaries=CLOSED), buoyancy_factor=0.1), physics=IncompressibleFlow())
        world.add(Inflow(Sphere(center=(5, 16, 16), radius=3), rate=0.2))


show()
