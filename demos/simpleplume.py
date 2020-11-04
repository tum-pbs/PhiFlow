from phi.flow import *


domain = Domain(x=64, y=80, boundaries=CLOSED, bounds=Box[0:100, 0:100])
buoyancy_factor = 0.1
inflow = domain.grid(Sphere(center=(50, 10), radius=5)) * 0.2
state = dict(velocity=domain.sgrid(0), density=domain.grid(0), pressure=domain.grid(0), divergence=domain.grid(0))


def step(velocity, density, pressure, divergence, dt):
    density = advect.semi_lagrangian(density, velocity, dt) + inflow
    velocity = advect.semi_lagrangian(velocity, velocity, dt) + (density * (0, buoyancy_factor)).at(velocity)
    velocity, pressure, iterations, divergence = fluid.make_incompressible(velocity, domain, relative_tolerance=1e-3, pressure_guess=pressure)
    return dict(velocity=velocity, density=density, pressure=pressure, divergence=divergence)


state = step(dt=5, **state)

app = App('Simple Plume', framerate=10)
app.set_state(state, step_function=step, dt=1.0, show=['velocity', 'density', 'divergence', 'pressure'])
app.add_field('Remaining Divergence', lambda: field.divergence(app.state['velocity']))
show(app, display=('density', 'velocity'))
