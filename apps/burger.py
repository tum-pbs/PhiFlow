from phi.model import *
from phi.flow import *


size = [32]
physics = Burger(Domain(size, STICKY), viscosity=0.1)


class Burger(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, "Burger's equation")
        self.value_velocity_scale = 2.0
        self.state = physics.random(rnd_scale = self.value_velocity_scale)

        self.add_field("Velocity", lambda: self.state)

    def step(self):
        self.state = physics.step(self.state)

    def action_reset(self, initial_res=3):
        self.state = physics.random(rnd_scale = self.value_velocity_scale)
        self.time = 0


app = Burger().show(framerate=4, production=__name__ != "__main__")
