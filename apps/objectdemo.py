from phi.flow import *


smoke = world.smoke(Domain([64, 64], SLIPPERY))
world.inflow(Sphere((10,32), 5), rate=0.2)


class MovingSphere(DynamicObject):

    def object_at(self, time):
        return Obstacle(Sphere([32, (time + 20) % 64], radius=5), SLIPPERY)


world.add(MovingSphere())


class Simpleplume(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, "Simpleplume", stride=5)
        self.add_field("Density", lambda: world.state[smoke].density)
        self.add_field("Velocity", lambda: world.state[smoke].velocity)
        self.add_field("Domain", lambda: smoke.domainstate.active(1))

    def step(self):
        world.step()


app = Simpleplume().show(production=__name__ != "__main__")
