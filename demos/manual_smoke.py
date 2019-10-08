from phi.flow import *






# advect density
# inflow density
# buoyancy
# pressure solve


# air1 = FLOW.step(air)


class ManualSmoke(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self)
        initial_velocity = math.zeros([1,81,65,2])
        initial_velocity[0,8:40, 30:34, 0] = 2
        self.air = StaggeredGrid('air', box[0:80, 0:64], initial_velocity, flags=[DIVERGENCE_FREE])
        self.add_field('Velocity', lambda: self.air.staggered_tensor())
        self.add_field('Divergence', lambda: self.air.divergence().data)
        self.add_field('Grad Div', lambda: StaggeredGrid.gradient(self.air.divergence()).staggered_tensor())
        self.add_field('Pressure', lambda: solve_pressure(self.air.divergence())[0].data)
        self.add_field('Grad Pressure', lambda: StaggeredGrid.gradient(solve_pressure(self.air.divergence())[0]).staggered_tensor())
        self.add_field('Div-free', lambda: divergence_free(self.air).staggered_tensor())
        self.add_field('Remaining div', lambda: divergence_free(self.air).divergence().data)
        self.value_dt = 1.0

    def step(self):
        self.air = divergence_free(self.air)
        self.air = advect.look_back(self.air, self.air, self.value_dt)



ManualSmoke().show()