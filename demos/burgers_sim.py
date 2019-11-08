from phi.flow import *


class BurgersDemo(App):

    def __init__(self, size=(64, 64)):
        App.__init__(self, "Burgers' equation", stride=5)
        self.value_velocity_scale = 2.0
        self.burgers1 = world.add(Burgers(Domain(size), self.initial_velocity, viscosity=0.1, periodic=True))
        self.burgers2 = world.add(Burgers(Domain(size), self.burgers1.velocity, viscosity=0.1, periodic=False))
        self.add_field('Periodic', lambda: self.burgers1.velocity)
        self.add_field('Boundaries', lambda: self.burgers2.velocity)

    def action_reset(self):
        self.steps = 0
        self.burgers1.velocity = self.initial_velocity
        self.burgers1.age = 0
        self.burgers2.velocity = self.burgers1.velocity
        self.burgers2.age = 0

    def initial_velocity(self, s): return math.randfreq(s) * self.value_velocity_scale


show(framerate=2)
# Equivalent to: app = BurgersDemo().show(framerate=2, production=__name__ != '__main__')
