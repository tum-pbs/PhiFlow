from unittest import TestCase

from phi.torch.flow import torch_from_numpy, World, Fluid, IncompressibleFlow, Obstacle, CLOSED, Inflow, Domain, Sphere, box


class TestFluidTF(TestCase):

    def test_fluid_tf(self):
        world = World()
        fluid = Fluid(Domain([16, 16], boundaries=CLOSED))
        fluid = torch_from_numpy(fluid)
        world.add(fluid, physics=IncompressibleFlow())
        world.add(Inflow(Sphere((8, 8), radius=4)))
        world.add(Obstacle(box[4:16, 0:8]))
        world.step()
        world.step()
