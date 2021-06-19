"""
Profiles the common fluid operations advection and pressure solve.
The profile is stored in the working directory and can be viewed with e.g. with Google chrome.
"""

from phi.physics._boundaries import Domain, STICKY as CLOSED
from phi.flow import *
# from phi.torch.flow import *
# from phi.tf.flow import *
# from phi.jax.flow import *


DOMAIN = Domain(x=128, y=128, boundaries=CLOSED, bounds=Box[0:100, 0:100])
velocity = DOMAIN.staggered_grid(Noise(vector=2))
pressure = DOMAIN.scalar_grid(0)

with backend.profile(save=f'navier_stokes_{backend.default_backend()}.json'):
    velocity = advect.semi_lagrangian(velocity, velocity, dt=1)
    # velocity, pressure = fluid.make_incompressible(velocity, DOMAIN, pressure_guess=pressure)
