from phi.flow import *


class HeatEquilibriumDemo(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, 'Heat Relaxation',
                                    'A horizontal plate at the top is heated and a sphere at the bottom is cooled.',
                                    stride=10)
        self.heat = world.Heat(Domain([64, 64]), diffusivity=1)
        world.HeatSource(box[44:46, 0:64], rate=1)
        world.ColdSource(Sphere([20, 32], 4), rate=1, physics=GeometryMovement(lambda t: Sphere([self.y, self.x], 4)))
        self.add_field('Temperature', lambda: self.heat.temperature)
        self.x, self.y = EditableInt('X', 32, (14,50)), EditableInt('Y', 20, (4,40))


HeatEquilibriumDemo().prepare().play(framerate=2).show(framerate=2)