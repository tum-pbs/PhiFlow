from phi.flow import *

adv = advect.semi_lagrangian

domain = Domain(x=64, y=80, boundaries=CLOSED, bounds=Box[0:100, 0:100])
inflow = domain.grid(Sphere(center=(50, 10), radius=5)) * 0.2
state = dict(velocity=domain.sgrid(0), density=domain.grid(0), pressure=domain.grid(0), divergence=domain.grid(0))


def step(velocity, density, pressure, divergence, dt):
    density = adv(density, velocity, dt) + inflow
    buoyancy_force = density * (0, 0.1) >> velocity
    velocity = adv(velocity, velocity, dt) + buoyancy_force
    velocity, pressure, iterations, divergence = fluid.make_incompressible(velocity, domain, pressure_guess=pressure)
    return dict(velocity=velocity, density=density, pressure=pressure, divergence=divergence)


state = step(dt=5, **state)

app = App('Simple Plume', framerate=10)
app.set_state(state, step_function=step, dt=1.0, show=['velocity', 'density', 'divergence', 'pressure'])
app.add_field('Remaining Divergence', lambda: field.divergence(app.state['velocity']))
show(app, display=('density', 'velocity'))
