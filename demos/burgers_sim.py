from phi.flow import *


domain = Domain(x=64, y=64, boundaries=PERIODIC, bounds=Box[0:100, 0:100])
state = dict(velocity=domain.sgrid(Noise(vector=2)) * 2)
# velocity = domain.vgrid(Noise(vector=2)) * 2
# velocity = domain.sgrid(Sphere([50, 50], radius=15)) * [2, 0] + [1, 0]

# velocity.at_centers()


def step(velocity: Grid, dt):
    velocity = field.diffuse(velocity, 0.1, dt)
    velocity = advect.semi_lagrangian(velocity, velocity, dt)
    write_sim_frame(app.directory, velocity, app.frame, 'velocity')
    return dict(velocity=velocity)


app = App('Burgers Equation in %dD' % len(domain.resolution), framerate=5)

state = step(dt=0, **state)

app.set_state(state, step, show=['velocity'])
show(app)
