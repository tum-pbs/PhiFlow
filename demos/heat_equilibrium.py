from phi.flow import *


DESCRIPTION = """
A horizontal plate at the top is heated and a sphere at the bottom is cooled.
Control the heat source using the sliders at the bottom.
"""

DOMAIN = Domain(x=64, y=64)


class HeatEquilibriumDemo(App):

    def __init__(self):
        App.__init__(self, 'Heat Relaxation', DESCRIPTION, framerate=30)
        self.set_state({'temperature': DOMAIN.scalar_grid(0)}, self.sim_step, show=['temperature'])
        self.x = EditableInt('X', 32, (14, 50))
        self.y = EditableInt('Y', 20, (4, 40))
        self.radius = EditableInt('Radius', 4, (2, 10))

    def sim_step(self, temperature, dt):
        temperature -= dt * DOMAIN.scalar_grid(Box[0:64, 44:46])
        temperature += dt * DOMAIN.scalar_grid(Sphere([self.x, self.y], radius=self.radius))
        return {'temperature': diffuse.explicit(temperature, 0.5, dt, substeps=4)}


show(autorun=True)
