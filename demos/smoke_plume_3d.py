""" Smoke Plume
Hot smoke is emitted from a circular region at the bottom.
The simulation computes the resulting air flow in a closed box.
"""
from phi.flow import *  # minimal dependencies
# from phi.torch.flow import *
# from phi.tf.flow import *
# from phi.jax.flow import *


velocity = StaggeredGrid((0, 0, 0), extrapolation.ZERO, x=32, y=32, z=32, bounds=Box(x=100, y=100, z=100))  # or CenteredGrid(...)
smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=32, y=32, z=32, bounds=Box(x=100, y=100, z=100))
INFLOW = 0.2 * CenteredGrid(SoftGeometryMask(Sphere(x=50, y=50, z=10, radius=5)), 0, smoke.bounds, smoke.resolution)
pressure = None


# @jit_compile  # Only for PyTorch, TensorFlow and Jax
def step(v, s, p, dt=1.):
    s = advect.mac_cormack(s, v, dt) + INFLOW
    buoyancy = s * (0, 0, 0.1) @ v  # resamples smoke to velocity sample points
    v = advect.semi_lagrangian(v, v, dt) + buoyancy * dt
    v, p = fluid.make_incompressible(v, (), Solve('auto', 1e-5, 0, x0=p))
    return v, s, p


for _ in view(smoke, velocity, 'pressure', play=False, namespace=globals()).range(warmup=1):
    velocity, smoke, pressure = step(velocity, smoke, pressure)
