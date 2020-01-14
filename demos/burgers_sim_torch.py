from phi.torch.flow import *


class BurgersEquation(App):

    def __init__(self, domain=Domain([64, 64], boundaries=PERIODIC)):
        App.__init__(self, framerate=5)
        velocity = world.add(DiffusiveVelocity(domain, velocity=lambda s: torch_from_numpy(math.randfreq(s) * 2)), physics=Burgers())
        self.add_field('Velocity', lambda: velocity.velocity.data.numpy())


show()
