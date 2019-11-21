from phi.flow import *


class BurgersDemo(App):

    def __init__(self, size=(64, 64)):
        App.__init__(self, "Burgers' equation", stride=5)
        self.value_velocity_scale = 2.0
        world.add(Burgers(Domain(size, boundaries=PERIODIC), self.initial_velocity, viscosity=0.1, name='burgers1'))
        world.add(Burgers(Domain(size, boundaries=OPEN), world.burgers1.velocity, viscosity=0.1, name='burgers2'))
        self.add_field('Periodic', lambda: world.burgers1.velocity)
        self.add_field('Boundaries', lambda: world.burgers2.velocity)

    def action_reset(self):
        self.steps = 0
        world.burgers1.velocity = self.initial_velocity
        world.burgers1.age = 0
        world.burgers2.velocity = world.burgers1.velocity
        world.burgers2.age = 0

    def initial_velocity(self, shape):
        return math.randfreq(shape) * self.value_velocity_scale


show(framerate=2)
