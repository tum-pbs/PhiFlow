from phi.flow import *


class Simpleplume(FieldSequenceModel):

    def __init__(self, domain=Domain([80, 64])):
        FieldSequenceModel.__init__(self, 'Simpleplume', stride=5)
        world.Gravity([-9.81, 0])
        air = world.DenseFluid(domain)
        temp = world.FluidProperty(domain, 'temperature')
        world.Buoyancy(temp, strength=0.2)
        # world.Fan(box[10, 30:34], [2, 0])
        # world.Inflow(Sphere((10, 32), 5), rate=0.2)
        world.HeatSource(Sphere((10, 32), 5), rate=0.2)
        self.add_field('Velocity', lambda: air.velocity)
        self.add_field('Temperature', lambda: temp.field)


app = Simpleplume().show(production=__name__ != '__main__')
