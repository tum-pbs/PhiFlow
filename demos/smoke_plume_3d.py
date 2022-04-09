""" Smoke Plume
Hot smoke is emitted from a circular region at the bottom.
The simulation computes the resulting air flow in a closed box.
"""

from phi.flow import *

DOMAIN = dict(x=32, y=32, z=32, bounds=Box(x=100, y=100, z=100))
velocity = StaggeredGrid(0, extrapolation.ZERO, **DOMAIN)  # or use CenteredGrid
smoke = CenteredGrid(0, extrapolation.BOUNDARY, **DOMAIN)
INFLOW = 0.2 * CenteredGrid(SoftGeometryMask(Sphere(x=50, y=50, z=10, radius=5)), extrapolation.ZERO, **DOMAIN)
pressure = None


def step(smoke, velocity, pressure):
    smoke = advect.mac_cormack(smoke, velocity, 1) + INFLOW
    buoyancy_force = smoke * (0, 0, 0.1) @ velocity  # resamples smoke to velocity sample points
    velocity = advect.semi_lagrangian(velocity, velocity, 1) + buoyancy_force
    velocity, pressure = fluid.make_incompressible(velocity, (), Solve('auto', 1e-3, 0, x0=pressure))
    return smoke, velocity, pressure


for _ in view(smoke, velocity, 'pressure', play=False, namespace=globals()).range(warmup=1):
    smoke, velocity, pressure = step(smoke, velocity, pressure)
