from phi.torch.flow import *


class Simpleplume(App):

    def __init__(self):
        App.__init__(self, framerate=10)
        fluid = world.add(Fluid(Domain([80, 64], boundaries=CLOSED), buoyancy_factor=0.1), physics=IncompressibleFlow())
        fluid.density = fluid.density.with_data(torch.from_numpy(fluid.density.data))
        world.add(Inflow(Sphere(center=(10, 32), radius=5), rate=0.2))
        self.add_field('Density', lambda: fluid.density.data.numpy())


show()
