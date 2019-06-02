from phi.tf.flow import *
from phi.model import *


class RandomSmoke(FieldSequenceModel):

    def __init__(self, size, steps=12):
        FieldSequenceModel.__init__(self, "Random Smoke Simulation",
                                    "PhiFlow demo showcasing complete smoke simulation within tensorflow",
                                    record_data=True,
                                    base_dir="SmokeIK",
                                    recorded_fields=("Density", "Velocity"),
                                    summary="random_"+"x".join([str(i) for i in size]),)
        self.add_field("Density", lambda: self.density_data)
        self.add_field("Velocity", lambda: self.velocity_data)

        self.sim = TFFluidSimulation(size, "open", 1)

        self.velocity_in = self.sim.placeholder("staggered", "velocity")
        self.density_in = self.sim.placeholder(1, "density")
        self.sim_step(self.velocity_in, self.density_in, self.sim)

        self.v_scale = EditableFloat("Velocity Base Scale", 1.0, (0, 4), log_scale=False)
        self.v_scale_rnd = EditableFloat("Velocity Randomization", 0.5, (0, 1), log_scale=False)
        self.v_falloff = EditableFloat("Velocity Power Spectrum Falloff", 0.9, (0, 1), log_scale=False)
        self.steps_per_scene = EditableInt("Steps per Scene", steps, (1, 32))
        self.prepare()
        self.sim.initialize_variables()
        self.action_reset(False)

    def sim_step(self, velocity, density, sim):
        self.density_out = velocity.advect(density)
        self.velocity_out = sim.divergence_free(velocity.advect(velocity) + sim.buoyancy(density))

    def step(self):
        self.velocity_data, self.density_data = self.sim.run([self.velocity_out, self.density_out], feed_dict=self.feed())
        self.info("Finished step %d of scene %s"%(self.time, self.scene))

    def feed(self):
        return {self.velocity_in: self.velocity_data, self.density_in: self.density_data}

    def action_reset(self, do_step=True):
        v_scale = self.v_scale * (1 + (np.random.rand()-0.5) * 2)
        self.info("Creating scene %s with v_scale=%f, v_falloff=%f"%(self.scene, v_scale, self.v_falloff))
        self.add_custom_properties({"velocity_scale": v_scale, "velocity_falloff": self.v_falloff})
        margin = int(round(v_scale * 16)) + 10

        # Velocity
        size = [1 for dim in self.sim.dimensions]
        rand = np.zeros([1]*(len(size)+1)+[len(size)])
        i = 0
        while size[0] < self.sim.dimensions[0]:
            rand = upsample2x(rand)
            size = [s * 2 for s in size]
            rand += np.random.randn(*([1]+size+[len(size)])) * v_scale * self.v_falloff**i
            i += 1
        rand = math.pad(rand, [[0,0]]+ [[0,1]]*self.sim.rank + [[0,0]], "symmetric")
        self.velocity_data = StaggeredGrid(rand)

        # Density
        density_data = upsample2x(upsample2x(np.random.rand(*self.sim.shape(scale=1.0/4)))) \
                            + upsample2x(upsample2x(upsample2x(np.random.rand(*self.sim.shape(scale=1.0/8))))) \
                            + upsample2x(upsample2x(upsample2x(upsample2x(np.random.rand(*self.sim.shape(scale=1.0/16))))))
        density_data = np.minimum(np.maximum(0, density_data * 0.66 - 1), 1)
        self.density_data = self.sim.zeros()
        valid_density_range = [slice(None)]+[slice(margin, -margin)]*self.sim.rank+[slice(None)]
        self.density_data[valid_density_range] = density_data[valid_density_range]

        self.info(0)
        if do_step: self.step()

    def action_next_scene(self, new_dir=True):
        self.action_reset()
        if new_dir:
            self.new_scene()
        self.play(self.steps_per_scene, callback=self.action_next_scene)
        return self


app = RandomSmoke([128] * 2).action_next_scene(new_dir=False).show(depth=31, production=__name__ != "__main__")
