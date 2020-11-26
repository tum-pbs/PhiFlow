from phi.flow import *


DESCRIPTION = """
Fluid simulation with additional marker fields that are passively transported with the fluid.

The dense marker is sampled on a regular grid while the sparse marker is a collection of particles.
"""


def checkerboard(size=8, offset=2):
    return math.to_float(math.all((domain.cells.center - offset) % (2 * size) < size, 'vector'))


domain = Domain([160, 126], CLOSED)
velocity = domain.sgrid(Noise(vector=2, scale=100)) * 4
dense_marker = CenteredGrid(checkerboard(), domain.bounds)
points = math.join_dimensions(domain.cells.center.x[::4].y[::4], ('x', 'y'), 'points').points.as_batch()
sparse_marker = PointCloud(Sphere(points, 1), math.random_normal(points.shape.without('vector')))
state = dict(velocity=velocity, markers=(dense_marker, sparse_marker))


def step(dt, velocity: CenteredGrid, markers: tuple):
    velocity = advect.semi_lagrangian(velocity, velocity, dt)
    velocity, _, _, _ = fluid.make_incompressible(velocity, domain)
    markers = [advect.advect(m, velocity, dt) for m in markers]
    return dict(velocity=velocity, markers=markers)


app = App('Passive Markers', DESCRIPTION, framerate=10, dt=0.2)
app.set_state(state, step, show=['velocity'])
app.add_field('Dense Marker', lambda: app.state['markers'][0])
app.add_field('Sparse Marker', lambda: domain.grid(app.state['markers'][1]))
show(app, display=('Dense Marker', 'Sparse Marker'))
