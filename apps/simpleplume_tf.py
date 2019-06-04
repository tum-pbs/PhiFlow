from phi.tf.model import *
from phi.flow import *
from phi.geom import *
from phi.tf.session import Session


size = [128]*2
smoke = Smoke(Domain(size, SLIPPERY))
obstacle(box[60:64, 40:128-40])
inflow(box[size[-2]//8, size[-1]*3//8:size[-1]*5//8])


class SimpleplumeNP(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, 'Simpleplume'+'x'.join([str(d) for d in size]), stride=3)
        self.session = Session(self.scene)
        self.current_state = SmokeState(tf.placeholder(tf.float32, smoke.domain.grid.shape(1)),
                                        StaggeredGrid(tf.placeholder(tf.float32, smoke.domain.grid.staggered_shape())))
        self.next_state = smoke.step(self.current_state)
        self.state = smoke.empty()

        self.add_field('Density', lambda: self.state.density)
        self.add_field('Velocity', lambda: self.state.velocity)
        self.add_field('Pressure', lambda: smoke.last_pressure)
        self.add_field('Divergence after', lambda: divergence(self.state.velocity))
        self.add_field('Domain', lambda: smoke.domainstate.active(extend=1))

        self.step()

    def step(self):
        self.state = self.session.run(self.next_state, {self.current_state: self.state})


app = SimpleplumeNP().show(display=('Density', 'Velocity'), framerate=2, production=__name__!='__main__')
