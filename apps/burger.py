from phi.flow import *


size = [64] * 2
dt = 1.0
physics = Burger(Domain(size, STICKY), viscosity=0.1 * dt)
physics.dt = dt


class BurgerApp(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, "Burger's equation")
        self.value_velocity_scale = 2.0
        self.state = randn(physics.shape(), [0, 0, self.value_velocity_scale])
        self.add_field("Velocity", lambda: self.state)

    def step(self):
        self.state = physics.step(self.state)

    def action_reset(self):
        self.state = randn(physics.shape(), [0, 0, self.value_velocity_scale])
        self.time = 0


app = BurgerApp().show(framerate=4, production=__name__ != "__main__")