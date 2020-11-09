from phi.flow import *


def step(dt, velocity, obstacle):
    obstacle = obstacle.copied_with(geometry=obstacle.geometry.rotated(- obstacle.angular_velocity * dt))  # rotate bar
    velocity = advect.semi_lagrangian(velocity, velocity, dt)
    velocity, pressure, iterations, divergence = fluid.make_incompressible(velocity, domain, (obstacle,))
    return dict(velocity=velocity, obstacle=obstacle)


domain = Domain(x=128, y=128, boundaries=OPEN, bounds=Box[0:100, 0:100])
state = dict(velocity=domain.sgrid((1, 0)), obstacle=Obstacle(Box[48:52, 10:90], angular_velocity=0.1))
state = step(1, **state)

app = App('Moving Objects Demo', framerate=10)
app.set_state(state, step, show=['velocity'])
app.add_field('Domain', lambda: domain.grid(app.state['obstacle'].geometry))
show(app, display=('Domain', 'velocity'))
