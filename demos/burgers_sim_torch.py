from phi.torch.flow import *


class BurgersEquation(App):

    def __init__(self, domain=Domain([64, 64])):
        App.__init__(self, framerate=5)
        initial_velocity = domain.centered_grid(data=lambda s: math.randfreq(s) * 2, components=domain.rank, name='velocity')
        velocity = world.add(initial_velocity, physics=Burgers(viscosity=0.1))
        velocity.data = torch.from_numpy(velocity.data)
        self.add_field('Velocity', lambda: velocity.data.numpy())


show()
