from phi.flow import *


class BurgerDemo(FieldSequenceModel):

    def __init__(self, size=(64, 64)):
        FieldSequenceModel.__init__(self, "Burger's equation")
        self.value_velocity_scale = 2.0
        self.burger = world.Burger(Domain(size), randn(levels=[0, 0, self.value_velocity_scale]), viscosity=0.1)
        self.add_field('Velocity', lambda: self.burger.velocity)

    def action_reset(self):
        self.burger.velocity = randn(levels=[0, 0, self.value_velocity_scale])
        self.burger.age = 0
        self.steps = 0

    def step(self):
        world.step(dt=0.2)
        self.info('Simulation time: %1f' % self.burger.age)


app = BurgerDemo().show(framerate=4, production=__name__ != '__main__')