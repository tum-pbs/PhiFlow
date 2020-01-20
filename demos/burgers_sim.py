from phi.flow import *


class BurgersEquation(App):

    def __init__(self, domain=Domain([64, 64], boundaries=PERIODIC)):
        App.__init__(self, framerate=5)
        world.add(BurgersVelocity(domain, velocity=lambda s: math.randfreq(s) * 2), physics=Burgers())


show()
