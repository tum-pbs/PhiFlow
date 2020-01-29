from phi.tf.flow import *


class BurgersEquation(App):

    def __init__(self, domain=Domain([2560, 256], boundaries=CLOSED)):
        App.__init__(self, framerate=5)
        world.add(BurgersVelocity(domain, velocity=lambda s: math.randfreq(s) * 2), physics=Burgers())


show()
