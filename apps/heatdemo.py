from phi.flow import *


class HeatDemo(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, 'Heat Demo', stride=100)
        self.heat = world.Heat(Domain([64, 64]), diffusivity=0.2)
        world.HeatSource(box[30:40, 10:12], 0.1)
        world.ColdSource(box[30:40, 40:42], 0.1)
        self.add_field('Temperature', lambda: self.heat.temperature)


HeatDemo().show()