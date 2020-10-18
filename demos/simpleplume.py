from phi.flow import *


domain = Domain([128, 160], boundaries=CLOSED, box=Box[0:100, 0:100])
dt = 1.0
buoyancy_factor = 0.1

velocity = domain.sgrid(0)
density = domain.grid(Sphere(center=(50, 10), radius=5))
inflow = density * 0.2
pressure = domain.grid(0)
divergence = domain.grid(0)


def step():
    global velocity, density, divergence, pressure
    density = advect.semi_lagrangian(density, velocity, dt) + inflow
    velocity = advect.semi_lagrangian(velocity, velocity, dt) + (density * (0, buoyancy_factor)).at(velocity)
    velocity, pressure, iterations, divergence = field.divergence_free(velocity, bake=None, relative_tolerance=0.01)


step()


app = App('Simple Plume', framerate=10)
app.add_field('Velocity', lambda: velocity)
app.add_field('Density', lambda: density)
app.add_field('Divergence', lambda: divergence)
app.add_field('Pressure', lambda: pressure)
app.add_field('Inflow', lambda: inflow)
app.step = step
show(app)

# while True:
#
#     app.update({'velocity': velocity, 'density': density})
