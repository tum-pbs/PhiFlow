""" Smoke Plume
Hot smoke is emitted from a circular region at the bottom.
The simulation computes the resulting air flow in a closed box.
"""

from phi.flow import *

DOMAIN = Domain(x=80, y=80, boundaries=CLOSED, bounds=Box[0:100, 0:100])
INFLOW = DOMAIN.scalar_grid(Sphere(center=(50, 10), radius=5)) * 0.2
velocity = DOMAIN.staggered_grid(0)  # alternatively vector_grid(0)
smoke = DOMAIN.scalar_grid(0)
pressure = DOMAIN.scalar_grid(0)
divergence = DOMAIN.scalar_grid(0)

for _ in view(smoke, velocity, pressure, divergence, play=False).range():
    smoke = advect.mac_cormack(smoke, velocity, 1) + INFLOW
    buoyancy_force = smoke * (0, 0.1) >> velocity  # resamples smoke to velocity sample points
    velocity = advect.semi_lagrangian(velocity, velocity, 1) + buoyancy_force
    velocity, pressure_result = fluid.make_incompressible(velocity, DOMAIN, solve=Solve('CG-adaptive', 1e-5, 0, x0=pressure))
    pressure = pressure_result.x
