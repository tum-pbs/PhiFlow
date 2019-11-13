import sys
if 'tf' in sys.argv:
    from phi.tf.flow import *  # Use TensorFlow
    MODE = 'TensorFlow'
else:
    from phi.flow import *  # Use NumPy
    MODE = 'NumPy'


def create_tum_logo():
    for x in range(1, 10, 2):
        world.add(Obstacle(box[41:83, 15 + x * 7:15 + (x+1) * 7]))
    world.add_all(Obstacle(box[41:48, 43:50]), Obstacle(box[83:90, 15:43]), Obstacle(box[83:90, 50:85]))


class SmokeLogo(App):

    def __init__(self, size):
        App.__init__(self, 'Smoke Logo', 'Run a smoke simulation using %s for processing.' % MODE,
                     summary='smokedemo' + 'x'.join([str(d) for d in size]), stride=20)
        smoke = self.smoke = world.add(Smoke(Domain(size, box=box[0:100, 0:100], boundaries=SLIPPERY)))
        world.add_all(Inflow(box[6:10, 14:21], rate=1.0), Inflow(box[6:10, 79:86], 0.8), Inflow(box[49:50, 43:46], 0.1))
        create_tum_logo()
        # Add Fields
        self.add_field('Density', lambda: smoke.density)
        self.add_field('Velocity', lambda: smoke.velocity)
        self.add_field('Domain', lambda: obstacle_mask(smoke).at(smoke.density))
        self.add_field('Remaining Divergence', lambda: smoke.velocity.divergence())

    def action_reset(self):
        self.steps = 0
        self.smoke.density = self.smoke.velocity = 0


show(SmokeLogo([int(sys.argv[1])] * 2 if len(sys.argv) > 1 and __name__ == '__main__' else [128] * 2),
     display=('Density', 'Velocity'), framerate=2)
