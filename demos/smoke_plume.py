""" Smoke Plume
Hot smoke is emitted from a circular region at the bottom.
The simulation computes the resulting air flow in a closed box.
"""

from phi.flow import *

DOMAIN = Domain(x=80, y=80, boundaries=CLOSED, bounds=Box[0:100, 0:100])
INFLOW = DOMAIN.scalar_grid(Sphere(center=(50, 10), radius=5)) * 0.2
velocity = DOMAIN.staggered_grid(0)  # alternatively vector_grid(0)
smoke = pressure = divergence = DOMAIN.scalar_grid(0)

for _ in ModuleViewer(display=('smoke', 'velocity')).range():
    smoke = advect.semi_lagrangian(smoke, velocity, 1) + INFLOW
    buoyancy_force = smoke * (0, 0.1) >> velocity  # resamples smoke to velocity sample points
    velocity = advect.semi_lagrangian(velocity, velocity, 1) + buoyancy_force
    velocity, pressure, iterations, divergence = fluid.make_incompressible(velocity, DOMAIN, pressure_guess=pressure)
