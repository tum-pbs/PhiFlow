from unittest import TestCase

from phi.torch.flow import torch_from_numpy, World, Fluid, IncompressibleFlow, Obstacle, CLOSED, Inflow, Domain, Sphere, box, OPEN, STICKY, SLIPPERY, PERIODIC, Noise, struct, numpy, math


class TestFluidPyTorch(TestCase):

    def test_fluid_pytorch(self):
        world = World()
        fluid = Fluid(Domain([16, 16], boundaries=CLOSED))
        fluid = torch_from_numpy(fluid)
        world.add(fluid, physics=IncompressibleFlow())
        world.add(Inflow(Sphere((8, 8), radius=4)))
        world.add(Obstacle(box[4:16, 0:8]))
        world.step()
        world.step()

    def test_fluid_pytorch_equality(self):
        for domain in [
            Domain([8, 6], boundaries=OPEN),
            Domain([8, 6], boundaries=STICKY),
            Domain([8, 6], boundaries=SLIPPERY),
            Domain([8, 6], boundaries=PERIODIC),
            Domain([8, 6], boundaries=[PERIODIC, [OPEN, STICKY]])
        ]:
            print('Comparing on domain %s' % (domain.boundaries,))
            np_fluid = Fluid(domain, density=Noise(), velocity=Noise(), batch_size=10)
            torch_fluid = math.to_float(torch_from_numpy(np_fluid))
            physics = IncompressibleFlow(conserve_density=False)
            for _ in range(3):
                np_fluid = physics.step(np_fluid, 1.0)
                torch_fluid = physics.step(torch_fluid, 1.0)
                for np_tensor, torch_tensor in zip(struct.flatten(np_fluid), struct.flatten(torch_fluid)):
                    torch_eval = torch_tensor.numpy()
                    numpy.testing.assert_almost_equal(np_tensor, torch_eval, decimal=5)

