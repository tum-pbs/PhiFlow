from phi.flow import *


class Simpleplume(App):

    def __init__(self):
        App.__init__(self, stride=5)
        world.add(Smoke(Domain([80, 64], boundaries=SLIPPERY)))
        world.add(Inflow(Sphere(center=(10, 32), radius=5), rate=0.2))


show()
