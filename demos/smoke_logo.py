import sys
if 'tf' in sys.argv:
    from phi.tf.flow import *  # Use TensorFlow
    mode = 'TensorFlow'
else:
    from phi.flow import *  # Use NumPy
    mode = 'NumPy'


def build_inflow(x, y, world=world):
    dx, dy = x * 0.07, y * 0.07
    offset_x = 0.5 * (x - dx * 10)
    offset_y = y - y * 0.1 - dy * 7
    world.Inflow(box[y/16, 2*dx:3*dx])
    world.Inflow(box[y/16, x-3*dx:x-2*dx])
    world.Inflow(box[offset_y+dy+1, offset_x+4*dx:offset_x+4.5*dx], 0.1)


def create_tum_logo(x, y, world=world):
    dx, dy = x * 0.07, y * 0.07
    offset_x = 0.5 * (x - dx * 10)
    offset_y = y - y * 0.1 - dy * 7
    for i in range(1, 10, 2):
        world.Obstacle(box[offset_y:offset_y+6*dy, offset_x + i * dx:offset_x + (i+1) * dx])
    world.Obstacle(box[offset_y:offset_y+dy, offset_x + 4 * dx:offset_x + 4 * dx+ dx])
    world.Obstacle(box[offset_y + 6 * dy:offset_y + 7 * dy, offset_x:offset_x+dx * 4])
    world.Obstacle(box[offset_y + 6 * dy:offset_y + 7 * dy, offset_x + 5 * dx:offset_x + 10 * dx])


class SmokeLogo(FieldSequenceModel):

    def __init__(self, size):
        FieldSequenceModel.__init__(self, 'Smoke Demo','Run a smoke simulation using %s for processing.' % mode,
                         summary='smokedemo' + 'x'.join([str(d) for d in size]), stride=20)
        smoke = self.smoke = world.Smoke(Domain(size, SLIPPERY))
        build_inflow(*size)
        create_tum_logo(*size)
        self.add_field('Density', lambda: smoke.density)
        self.add_field('Velocity', lambda: smoke.velocity)
        self.add_field('Domain', lambda: smoke.domaincache.active(extend=1))
        self.add_field('Pressure', lambda: smoke.last_pressure)
        self.add_field('Remaining Divergence', lambda: divergence(smoke.velocity))

    def action_reset(self):
        self.steps = 0
        self.smoke.density = self.smoke.velocity = 0


app = SmokeLogo([int(sys.argv[1])] * 2 if len(sys.argv) > 1 else [128] * 2)\
    .show(display=('Density', 'Velocity'), framerate=2, production=__name__ != '__main__')