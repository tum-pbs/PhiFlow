import random
from phi.tf.model import *
from phi.flow import *
from phi.geom import *


class SimpleplumeNP(FieldSequenceModel):

    def __init__(self, size=(128, 128), solver=None):
        FieldSequenceModel.__init__(self, 'Simpleplume'+'x'.join([str(d) for d in size]), stride=3)

        session = tf.Session()

        self.smoke = Smoke(Domain(size, SLIPPERY), pressure_solver=solver)
        load_tensorflow()
        self.smoke.density = tf.placeholder(tf.float32, self.smoke.domain.grid.shape(1))
        self.smoke.velocity.staggered = tf.placeholder(tf.float32, self.smoke.domain.grid.staggered_shape())

        obstacle(box[60:64, 40:128-40])
        inflow(box[size[-2]//8, size[-1]*3//8:size[-1]*5//8])

        session.run(tf.global_variables_initializer())

        self.add_field('Density', lambda: self.smoke.density)
        self.add_field('Velocity', lambda: self.smoke.velocity)
        self.add_field('Pressure', lambda: self.smoke.last_pressure)
        self.add_field('Divergence after', lambda: divergence(self.smoke.velocity))
        self.add_field('Domain', lambda: self.smoke.domain.active(extend=1))

        self.step()

    def step(self):
        self.smoke.step(1)


app = SimpleplumeNP().show(display=('Density', 'Velocity'), framerate=2, production=__name__!='__main__')
