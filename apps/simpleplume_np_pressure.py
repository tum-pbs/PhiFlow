import random
from phi.model import *
from phi.flow import *
from phi.solver.sparse import SparseCG


class SimpleplumeNP(FieldSequenceModel):

    def __init__(self, size=(128, 128), solver=None):
        sim = FluidSimulation(size, 'closed', solver=solver, buoyancy_factor=0.1)
        # print(sim.solver.solvers[-1].accuracy)
        FieldSequenceModel.__init__(self, 'Simpleplume'+'x'.join([str(d) for d in size]),
                                    'Smoke simulation with NumPy and %s solver.' % sim.solver)
        self.sim = sim
        self.inflow_density = sim.zeros()
        self.inflow_density[..., size[-2]//8, size[-1]*3//8:size[-1]*5//8, :] = 1
        # self.sim.set_obstacle(10, (60, 20))
        self.action_reset()

        self.add_field('Density', lambda: self.density)
        self.add_field('Velocity', lambda: self.velocity)
        self.add_field('Pressure', lambda: sim.last_pressure)
        self.add_field('Advected Velocity', lambda: self.advected_velocity)
        self.add_field('Divergence before', lambda: divergence(self.advected_velocity))
        self.add_field('Divergence after', lambda: divergence(self.velocity))
        self.add_field('Domain', lambda: sim.extended_fluid_mask)
        self.data = []

    def step(self):
        self.density = normalize_to(self.velocity.advect(self.density), self.density) + self.inflow_density
        self.advected_velocity = self.velocity.advect(self.velocity)
        self.velocity = self.sim.divergence_free(self.advected_velocity + self.sim.buoyancy(self.density))
        self.info(self.sim.last_iter)
        self.data.append(self.sim.last_iter)

    def action_reset(self):
        self.velocity = self.sim.zeros('velocity')
        self.density = self.inflow_density
        self.time = 0

    def action_save(self):
        np.savetxt(self.scene.subpath("iter.txt"), self.data)


app = SimpleplumeNP()
app.play(200, callback=app.action_save)
exit(0)
# app.show(display=("Density", "Pressure"))