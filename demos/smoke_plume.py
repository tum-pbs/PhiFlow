from phi.flow import *

# tf.compat.v1.disable_eager_execution()

DOMAIN = Domain(x=80, y=80, boundaries=CLOSED, bounds=Box[0:100, 0:100])
inflow = DOMAIN.grid(Sphere(center=(50, 10), radius=5)) * 0.2
# velocity = DOMAIN.vgrid(0)
velocity = DOMAIN.sgrid(0)
smoke = pressure = divergence = remaining_divergence = DOMAIN.grid(0)

for _ in range(20):
# for _ in ModuleViewer(port=8050).range():
    smoke = advect.semi_lagrangian(smoke, velocity, 1) + inflow
    buoyancy_force = smoke * (0, 0.1) >> velocity  # resamples smoke to velocity sample points
    velocity = advect.semi_lagrangian(velocity, velocity, 1) + buoyancy_force
    velocity, pressure, iterations, divergence = fluid.make_incompressible(velocity, DOMAIN, pressure_guess=pressure)
    remaining_divergence = field.divergence(velocity)
