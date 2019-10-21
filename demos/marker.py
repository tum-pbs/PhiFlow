from phi.flow import *


def checkerboard(resolution, size=4):
    data = math.zeros([1]+list(resolution)+[1])
    for y in range(size):
        for x in range(size):
            data[:, y+1::size*2, x+1::size*2, :] = 1
    return data


class MarkerDemo(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, 'Passive Marker', 'Simpleplume simulation + marker field', stride=5)
        smoke = self.smoke = world.Smoke(Domain([80, 64], SLIPPERY))
        self.marker = smoke.density.copied_with(data=checkerboard(smoke.domain.resolution))
        world.Inflow(Sphere((10, 32), 5), rate=0.2)
        self.add_field('Density', lambda: smoke.density)
        self.add_field('Velocity', lambda: smoke.velocity)
        self.add_field('Marker', lambda: self.marker)

    def step(self):
        world.step()
        self.marker = advect.look_back(self.marker, self.smoke.velocity, 1)


app = MarkerDemo().show(production=__name__ != '__main__')
