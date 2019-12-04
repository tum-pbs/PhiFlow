from phi.flow import *


class BurgersEquation(App):

    def __init__(self, domain=Domain([64, 64], boundaries=PERIODIC)):
        App.__init__(self, stride=5)
        velocity = world.add(domain.centered_grid(lambda s: math.randfreq(s) * 2, components=domain.rank, name='v'),
                             physics=Burgers(viscosity=0.1))
        self.add_field('Velocity', velocity)


show(framerate=2)
