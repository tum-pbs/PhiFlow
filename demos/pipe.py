from phi.flow import *


domain = Domain(x=50, y=32, boundaries=[OPEN, STICKY])
boundary_mask = HardGeometryMask(Box[:0.5, :]).at(domain.sgrid())
state = dict(velocity=domain.sgrid())


def step(dt, velocity):
    velocity = advect.semi_lagrangian(velocity, velocity, dt)
    velocity = velocity * (1 - boundary_mask) + boundary_mask * (1, 0)
    velocity, pressure, iterations, _ = fluid.make_incompressible(velocity, domain)
    velocity = field.diffuse(velocity, 0.1, dt)
    return dict(velocity=velocity)


state = step(1, **state)


app = App('Streamline Profile', 'Vertical Pipe')
app.set_state(state, step, show=['velocity'])
app.prepare()
app.step()
show(app)
