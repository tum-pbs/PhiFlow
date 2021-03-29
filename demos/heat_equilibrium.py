from phi.flow import *


DESCRIPTION = """
A horizontal plate at the top is heated and a sphere at the bottom is cooled.
Control the heat source using the sliders at the bottom.
"""

DOMAIN = Domain(x=64, y=64)
DT = 1.0


class HeatEquilibriumDemo(App):

    def __init__(self):
        App.__init__(self, 'Heat Relaxation', DESCRIPTION)
        self.temperature = DOMAIN.scalar_grid(0)
        self.add_field('temperature', lambda: self.temperature)
        self.x = EditableInt('X', 32, (14, 50))
        self.y = EditableInt('Y', 20, (4, 40))
        self.radius = EditableInt('Radius', 4, (2, 10))

    def step(self):
        self.temperature -= DT * DOMAIN.scalar_grid(Box[0:64, 44:46])
        self.temperature += DT * DOMAIN.scalar_grid(Sphere([self.x, self.y], radius=self.radius))
        return {'temperature': diffuse.explicit(self.temperature, 0.5, DT, substeps=4)}


show(HeatEquilibriumDemo(), framerate=30)
