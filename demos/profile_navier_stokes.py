"""
Profiles the common fluid operations advection and pressure solve.
The profile is stored in the working directory and can be viewed with e.g. with Google chrome.
"""

# Use one of the imports below to choose which backend should be used.

# NumPy / SciPy
from phi.flow import *

# PyTorch
# from phi.torch.flow import *

# TensorFlow
# from phi.tf.flow import *
# tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)  # prevent Blas GEMM launch failed


DOMAIN = Domain(x=128, y=128, boundaries=CLOSED, bounds=Box[0:100, 0:100])
velocity = DOMAIN.staggered_grid(Noise(vector=2))
pressure = DOMAIN.grid(0)

with math.backend.profile() as prof:
    for _ in range(2):
        velocity = advect.semi_lagrangian(velocity, velocity, dt=1)
        velocity, pressure, _, _ = fluid.make_incompressible(velocity, DOMAIN, pressure_guess=pressure)

prof.save_trace(f'navier_stokes_{math.backend.default_backend()}.json')
prof.print(min_duration=1e-2)
