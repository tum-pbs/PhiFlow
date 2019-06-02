from phi.model import *
from phi.flow import *


class Burger(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, "Burger's equation")
        self.sim = FluidSimulation([128], "open")
        self.add_field("Velocity", lambda: self.velocity)
        self.value_viscosity = 0.1
        self.value_velocity_scale = 2.0
        self.action_reset()

    def step(self):
        self.velocity = advect(self.velocity, self.velocity)
        self.velocity = self.velocity + self.value_viscosity * vector_laplace(self.velocity)

    def action_reset(self, initial_res=3):
        self.velocity = np.random.randn(*self.sim.shape("vector", scale=1.0 / 2 ** initial_res)) * self.value_velocity_scale
        for i in range(initial_res):
            self.velocity = upsample2x(self.velocity)
        self.time = 0


def vector_laplace(v):
    return np.concatenate([laplace(v[...,i:i+1]) for i in range(v.shape[-1])], -1)


app = Burger().show(framerate=4, production=__name__ != "__main__")
