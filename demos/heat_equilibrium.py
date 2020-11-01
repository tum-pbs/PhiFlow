from phi.flow import *


DESCRIPTION = """
A horizontal plate at the top is heated and a sphere at the bottom is cooled.
"""

DOMAIN = Domain([64, 64])


class HeatEquilibriumDemo(App):

    def __init__(self):
        App.__init__(self, 'Heat Relaxation', DESCRIPTION, framerate=30)
        self.set_state({'temperature': DOMAIN.grid(0)}, self.sim_step, show=['temperature'])
        self.x, self.y, self.rad = EditableInt('X', 32, (14, 50)), EditableInt('Y', 20, (4, 40)), EditableInt('Radius', 4, (2, 10))

    def sim_step(self, temperature, dt):
        temperature -= dt * DOMAIN.grid(Box[0:64, 44:46])
        temperature += dt * DOMAIN.grid(Sphere([self.x, self.y], radius=self.rad))
        return {'temperature': field.diffuse(temperature, 0.5, dt, substeps=4)}


display.AUTORUN = True
show()
