from phi.flow import *






# advect density
# inflow density
# buoyancy
# pressure solve


# air1 = FLOW.step(air)


class ManualSmoke(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self)
        initial_velocity = math.zeros([1,80,64,2])
        initial_velocity[0,8:12, 30:34, 0] = 0.1
        self.air = StaggeredGrid('air', box[0:4, 0:4], initial_velocity, flags=[DIVERGENCE_FREE])
        self.add_field('Velocity', lambda: self.air.staggered_tensor())
        self.value_dt = 0.1

    def step(self):
        self.air = advect.look_back(self.air, self.air, self.value_dt)
        self.pressure = solve_pressure(self.air)



ManualSmoke().show()