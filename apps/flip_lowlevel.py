from phi.model import FieldSequenceModel
from phi.flow import *
from phi.math.sampled import *

# If multiple batches are used, all simulations must have the same number of particles (can change over time)
# Outflows or dynamic spawns/destructions can only work with single-batch simulations

class FlipTest(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, "FLIP Test")
        sim = self.sim = FluidSimulation([30, 30])
        fluid_grid = sim.zeros()
        fluid_grid[:, 10:20, 10:20, :] = 1
        inflow_grid = sim.zeros()

        self.density = active_centers(fluid_grid)  # (1, n, 2) tensor
        self.inflow_density = active_centers(inflow_grid)  # (1, n, 2) tensor
        self.velocity = math.zeros_like(self.density)
        self.velocity += 1
        self.density += 8.1

        self.add_field("Density", lambda: grid(self.sim.griddef, self.density))
        self.add_field("Velocity", lambda: grid(self.sim.griddef, self.density, self.velocity, 0))

    def step_old(self):
        self.density = math.concat([self.density + self.velocity, self.inflow_density], axis=1)
        self.velocity = math.concat([self.velocity, math.zeros_like(self.inflow_density)], axis=1)
        self.velocity = self.sim.divergence_free_sampled(self.density, self.velocity + self.sim.gravity())


FlipTest().show()


# Classical simulation
# density = sim.zeros()
# inflow_density = sim.zeros()
# inflow_density[0, 8, 22:64-22, 0] = 1
# velocity = sim.zeros("velocity")  # why not density.zeros(...)?
#
# density = sim.advect(density, velocity) + inflow_density
# velocity = sim.divergence_free(sim.advect(velocity, velocity) + sim.gravity())
# velocity needs to be advected, too
# maybe make velocity a property of density so it automatically gets advected? Then density needs to be a new type.

# Fundamental difference: velocity needs to be advected in eulerian simulation but not in lagrangian