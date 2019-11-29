from phi.flow import *


def checkerboard(resolution, size=8, offset=2):
    data = math.zeros([1]+list(resolution)+[1])
    for y in range(size):
        for x in range(size):
            data[:, y+offset::size*2, x+offset::size*2, :] = 1
    return data


class MarkerDemo(App):

    def __init__(self):
        App.__init__(self, 'Passive Marker', 
            'Fluid simulation with a marker field and modified simulation step function', 
            stride=5)
        smoke = self.smoke = world.add(Smoke(Domain([160, 126], SLIPPERY)))
        self.marker = smoke.density.with_data(checkerboard(smoke.domain.resolution)) # the new field
        world.add(Inflow(Sphere((18, 64), 10), rate=0.2))
        self.add_field('Density', lambda: smoke.density)
        self.add_field('Velocity', lambda: smoke.velocity)
        self.add_field('Marker', lambda: self.marker)

    def step(self):
        world.step()
        # run an additional advection step to update the marker field
        self.marker = advect.semi_lagrangian(self.marker, self.smoke.velocity, 1)


show()
