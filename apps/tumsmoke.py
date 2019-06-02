from phi.tf.model import *
import sys


def build_inflow(sim):
    size = sim.dimensions
    le_x = int(math.floor(size[1] * 0.7 / 10))
    le_y = int(math.floor(size[0] * 0.7 / 10))
    offset_x = int(0.5 * math.floor(size[1] - le_x * 10))
    offset_y = int(size[0] - math.floor(size[0] * 0.1) - (le_y * 7))
    inflow_density = sim.zeros()
    inflow_density[:, size[0] // 16, 2*le_x:3*le_x] = 1
    inflow_density[:, size[0] // 16, size[1]-3*le_x:size[1]-2*le_x] = 1
    inflow_density[:, offset_y+le_y+1, offset_x+4*le_x:offset_x+4*le_x+le_x//2, :] = 0.1
    return inflow_density


def create_tum_logo(sim):
    size = sim.dimensions
    le_x = int(math.floor(size[1] * 0.7 / 10))
    le_y = int(math.floor(size[0] * 0.7 / 10))
    offset_x = int(0.5 * math.floor(size[1] - le_x * 10))
    offset_y = int(size[0] - math.floor(size[0] * 0.1) - (le_y * 7))
    for i in range(1, 10, 2):
        sim.set_obstacle((le_y * 6, le_x), (offset_y, offset_x + i * le_x))
    sim.set_obstacle((le_y, le_x), (offset_y, offset_x + 4 * le_x))
    sim.set_obstacle((le_y, le_x * 4), (offset_y + 6 * le_y, offset_x))
    sim.set_obstacle((le_y, le_x * 5), (offset_y + 6 * le_y, offset_x + 5 * le_x))


class SimpleplumeTF(TFModel):

    def __init__(self, size):
        TFModel.__init__(self, "TUMsmoke", "Smoke simulation with obstacles",
                         summary="TUMsmoke" + "x".join([str(d) for d in size]), stride=20)
        sim = self.sim = TFFluidSimulation(size, "closed", force_use_masks=True)
        velocity = self.velocity_in = self.sim.placeholder("staggered", "velocity")
        density = self.density_in = self.sim.placeholder(1, "density")
        self.inflow_density = build_inflow(self.sim)
        self.action_reset()
        self.density_out = sim.conserve_mass(velocity.advect(density), density) + self.inflow_density
        self.velocity_out = sim.divergence_free(velocity.advect(velocity) + sim.buoyancy(self.density_out))
        self.finalize_setup([])
        create_tum_logo(self.sim)

        self.add_field("Density", lambda: self.density_data)
        self.add_field("Velocity", lambda: self.velocity_data)
        self.add_field("Remaining Divergence", lambda: self.velocity_data.divergence())
        self.add_field("Obstacles", lambda: self.sim.run(self.sim.extended_active_mask))

    def step(self):
        self.velocity_data, self.density_data = self.sim.run([self.velocity_out, self.density_out],
                feed_dict={self.velocity_in.staggered: self.velocity_data, self.density_in: self.density_data})

    def action_reset(self):
        self.time = 0
        self.velocity_data = self.sim.zeros("staggered")
        self.density_data = self.inflow_density


size = 128
if len(sys.argv) > 1: size = int(sys.argv[1])
app = SimpleplumeTF([size]*2).show(display=("Density", "Velocity"), framerate=2, production=__name__ != "__main__")
