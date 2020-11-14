from phi.flow import *

domain = Domain(x=64, y=80, boundaries=CLOSED, bounds=Box[0:100, 0:100])
inflow = domain.grid(Sphere(center=(50, 10), radius=5)) * 0.2
velocity = domain.staggered_grid(0)
density = pressure = divergence = domain.grid(0)

for _ in ModuleViewer(framerate=20).range():
    density = advect.semi_lagrangian(density, velocity, 1) + inflow
    buoyancy_force = density * (0, 0.1) >> velocity
    velocity = advect.semi_lagrangian(velocity, velocity, 1) + buoyancy_force
    velocity, pressure, iterations, divergence = fluid.make_incompressible(velocity, domain, pressure_guess=pressure)
