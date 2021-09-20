""" Smoke Plume
Hot smoke is emitted from a circular region at the bottom.
The simulation computes the resulting air flow in a closed box.
"""

from phi.flow import *


DOMAIN = dict(x=64, y=64, bounds=Box[0:100, 0:100])
velocity = StaggeredGrid((0, 0), extrapolation.ZERO, **DOMAIN)  # or use CenteredGrid
smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=200, y=200, bounds=DOMAIN['bounds'])
INFLOW = 0.2 * CenteredGrid(SoftGeometryMask(Sphere(center=(50, 10), radius=5)), extrapolation.ZERO, resolution=smoke.resolution, bounds=smoke.bounds)
pressure = None

for _ in view(smoke, velocity, 'pressure', play=False, namespace=globals()).range(warmup=1):
    smoke = advect.mac_cormack(smoke, velocity, 1) + INFLOW
    buoyancy_force = smoke * (0, 0.1) @ velocity  # resamples smoke to velocity sample points
    velocity = advect.semi_lagrangian(velocity, velocity, 1) + buoyancy_force
    velocity, pressure = fluid.make_incompressible(velocity, (), Solve('auto', 1e-5, 0, x0=pressure))
