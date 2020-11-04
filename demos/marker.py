from phi.flow import *


DESCRIPTION = """
Fluid simulation with additional marker fields that are passively transported with the fluid.

The dense marker is sampled on a regular grid while the sparse marker is a collection of particles.
"""


def checkerboard(resolution: math.Shape, size=8, offset=2):
    data = math.zeros(resolution).numpy()
    for y in range(size):
        for x in range(size):
            data[y + offset::size * 2, x + offset::size * 2] = 1
    return math.tensor(data, names='x, y')


domain = Domain([160, 126], CLOSED)
velocity = domain.sgrid(Noise(vector=2))
dense_marker = CenteredGrid(checkerboard(domain.resolution), domain.bounds)
sparse_marker = PointCloud(reshape(domain.cells, '(x, y) -> points'))
state = dict(velocity=velocity, markers=(dense_marker, sparse_marker))


def step(velocity: CenteredGrid, markers: tuple):

    return dict(velocity=velocity, markers=markers)


app = App('Passive Markers', DESCRIPTION, framerate=10, dt=0.2)
app.set_state(state, step, show=['velocity'])
app.add_field('Dense Marker', lambda: app.state['markers'][0])
app.add_field('Sparse Marker', lambda: app.state['markers'][1].volume_sample(domain.cells))
show(app)
