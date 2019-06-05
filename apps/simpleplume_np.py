from phi.model import *
from phi.flow import *
from phi.geom import *


size = [128]*2
physics = Smoke(Domain(size, SLIPPERY))
obstacle(box[60:64, 40:128-40])
inflow(box[size[-2]//8, size[-1]*3//8:size[-1]*5//8])


class SimpleplumeNP(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, 'Simpleplume'+'x'.join([str(d) for d in size]), stride=3)
        self.state = physics.empty()

        self.add_field('Density', lambda: self.state.density)
        self.add_field('Velocity', lambda: self.state.velocity)
        self.add_field('Pressure', lambda: physics.last_pressure)
        self.add_field('Divergence after', lambda: divergence(self.state.velocity))
        self.add_field('Domain', lambda: physics.domainstate.active(extend=1))

    def step(self):
        self.state = physics.step(self.state)


app = SimpleplumeNP().show(display=('Density', 'Velocity'), framerate=2, production=__name__!='__main__')
