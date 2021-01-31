from phi.torch.flow import *


DOMAIN = Domain(x=128, y=128, boundaries=CLOSED, bounds=Box[0:100, 0:100])
velocity = DOMAIN.vgrid(Noise(vector=2))
pressure = DOMAIN.grid(0)

with math.backend.profile() as prof:
    for _ in range(2):
        velocity = advect.semi_lagrangian(velocity, velocity, dt=1)
        velocity, pressure, _, _ = fluid.make_incompressible(velocity, DOMAIN, pressure_guess=pressure)

prof.save_trace('navier_stokes.json')
prof.print(min_duration=1e-2, code_col=120)
