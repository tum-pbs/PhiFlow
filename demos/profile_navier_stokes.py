"""
Profiles the common fluid operations advection and pressure solve.
The profile is stored in the working directory and can be viewed with e.g. with Google chrome.
"""
from phi.flow import *
# from phi.torch.flow import *
# from phi.tf.flow import *
# from phi.jax.flow import *


velocity = StaggeredGrid((0, 0), 0, x=64, y=64, bounds=Box(x=100, y=100))

with backend.profile(save=f'navier_stokes_{backend.default_backend()}.json'):
    velocity = advect.semi_lagrangian(velocity, velocity, dt=1)
    # velocity, pressure = fluid.make_incompressible(velocity, DOMAIN, pressure_guess=pressure)
