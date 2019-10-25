import sys
if 'tf' in sys.argv:
    from phi.tf.flow import *  # Use TensorFlow
    mode = 'TensorFlow'
else:
    from phi.flow import *  # Use NumPy
    mode = 'NumPy'


def build_inflows():
    world.Inflow(box[6:10, 14:21], rate=1.0)
    world.Inflow(box[6:10, 79:86], rate=0.8)
    world.Inflow(box[49:50, 43:46], rate=0.1)


def create_tum_logo():
    for i in range(1, 10, 2):
        world.Obstacle(box[41:83, 15 + i * 7:15 + (i+1) * 7])
    world.Obstacle(box[41:48, 43:50])
    world.Obstacle(box[83:90, 15:43])
    world.Obstacle(box[83:90, 50:85])


class SmokeLogo(FieldSequenceModel):

    def __init__(self, size):
        FieldSequenceModel.__init__(self, 'Smoke Logo','Run a smoke simulation using %s for processing.' % mode,
                         summary='smokedemo' + 'x'.join([str(d) for d in size]), stride=20)
        smoke = self.smoke = world.Smoke(Domain(size, box=box[0:100, 0:100], boundaries=SLIPPERY))
        build_inflows()
        create_tum_logo()
        self.add_field('Density', lambda: smoke.density)
        self.add_field('Velocity', lambda: smoke.velocity)
        self.add_field('Domain', lambda: obstacle_mask(smoke).at(smoke.density))
        self.add_field('Pressure', lambda: smoke.last_pressure)
        self.add_field('Remaining Divergence', lambda: smoke.velocity.divergence())

    def action_reset(self):
        self.steps = 0
        self.smoke.density = self.smoke.velocity = 0


app = SmokeLogo([int(sys.argv[1])] * 2 if len(sys.argv) > 1 else [128] * 2)\
    .show(display=('Density', 'Velocity'), framerate=2, production=__name__ != '__main__')