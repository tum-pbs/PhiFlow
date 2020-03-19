from phi.flow import *


class Simpleplume(App):

    def __init__(self, domain=Domain([80, 64], boundaries=CLOSED)):
        App.__init__(self, framerate=10)
        world.add(*create_smoke(domain, buoyancy_factor=0.1))
        world.add(Inflow(Sphere(center=(10, 32), radius=5), rate=0.2))


show(Simpleplume())
