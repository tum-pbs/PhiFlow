from phi.tf.model import *
from phi.flow import *
import sys


size = [128]*2
if len(sys.argv) > 1:
    size = [int(sys.argv[1])]*2
smoke = Smoke(Domain(size, SLIPPERY))


def build_inflow():
    y, x = smoke.dimensions
    dx, dy = x * 0.07, y * 0.07
    offset_x = 0.5 * (x - dx * 10)
    offset_y = y - y * 0.1 - dy * 7
    inflow(box[y/16, 2*dx:3*dx])
    inflow(box[y/16, x-3*dx:x-2*dx])
    inflow(box[offset_y+dy+1, offset_x+4*dx:offset_x+4.5*dx], 0.1)


def create_tum_logo():
    y, x = smoke.dimensions
    dx, dy = x * 0.07, y * 0.07
    offset_x = 0.5 * (x - dx * 10)
    offset_y = y - y * 0.1 - dy * 7
    for i in range(1, 10, 2):
        obstacle(box[offset_y:offset_y+6*dy, offset_x + i * dx:offset_x + (i+1) * dx])
    obstacle(box[offset_y:offset_y+dy, offset_x + 4 * dx:offset_x + 4 * dx+ dx])
    obstacle(box[offset_y + 6 * dy:offset_y + 7 * dy, offset_x:offset_x+dx * 4])
    obstacle(box[offset_y + 6 * dy:offset_y + 7 * dy, offset_x + 5 * dx:offset_x + 10 * dx])


build_inflow()
create_tum_logo()


class SimpleplumeTF(TFModel):

    def __init__(self, size):
        TFModel.__init__(self, "TUMsmoke", "Smoke simulation with obstacles",
                         summary="TUMsmoke" + "x".join([str(d) for d in size]), stride=20)
        self.state = smoke.empty()
        self.add_field("Density", lambda: self.state.density)
        self.add_field("Velocity", lambda: self.state.velocity)
        self.add_field("Domain", lambda: smoke.domainstate.active(extend=1))

    def step(self):
        self.state = smoke.step(self.state)

    def action_reset(self):
        self.time = 0
        self.state = smoke.empty()


app = SimpleplumeTF([size]*2).show(display=("Density", "Velocity"), framerate=2, production=__name__ != "__main__")
