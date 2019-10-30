from phi.flow import *


class BurgersDemo(App):

    def __init__(self, size=(64, 64)):
        App.__init__(self, "Burgers' equation", stride=5)
        self.value_velocity_scale = 2.0
        self.burgers = world.add(Burgers(Domain(size), lambda s: math.randfreq(s) * self.value_velocity_scale, viscosity=0.1))
        self.add_field('Velocity', lambda: self.burgers.velocity)

    def action_reset(self):
        self.burgers.velocity = lambda s: math.randfreq(s) * self.value_velocity_scale
        self.burgers.age = 0
        self.steps = 0


show(framerate=2)
# Equivalent to: app = BurgersDemo().show(framerate=2, production=__name__ != '__main__')
