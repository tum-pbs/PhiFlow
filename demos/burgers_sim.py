from phi.tf.flow import *


class BurgersEquation(App):

    def __init__(self, domain=Domain([2560, 256], boundaries=CLOSED)):
        App.__init__(self, framerate=5)
        initial_velocity = domain.centered_grid(data=lambda s: math.randfreq(s) * 2, components=domain.rank, name='velocity')
        velocity = world.add(initial_velocity, physics=Burgers(viscosity=0.1))
        self.add_field('Velocity', velocity)


show()
