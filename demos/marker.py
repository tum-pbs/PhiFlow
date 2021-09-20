""" Passive Markers

Fluid simulation with additional marker fields that are passively transported with the fluid.

The dense marker is sampled on a regular grid while the sparse marker is a collection of particles.
"""

from phi.physics._boundaries import Domain, STICKY as CLOSED
from phi.flow import *


math.seed(0)


def checkerboard(size=8, offset=2):
    return math.to_float(math.all((DOMAIN.cells.center - offset) % (2 * size) < size, 'vector'))


DOMAIN = Domain(x=126, y=160, boundaries=CLOSED)
DT = 0.2

velocity = DOMAIN.staggered_grid(Noise(vector=2, scale=100)) * 4
dense_marker = CenteredGrid(checkerboard(), DOMAIN.boundaries['scalar'], DOMAIN.bounds)
points = math.pack_dims(DOMAIN.cells.center.x[::4].y[::4], ('x', 'y'), instance('points')).points.as_batch()
sparse_marker = DOMAIN.points(points)

for _ in view(framerate=10, play=False, namespace=globals()).range():
    velocity, _ = fluid.make_incompressible(velocity)
    dense_marker = advect.advect(dense_marker, velocity, DT)
    sparse_marker = advect.advect(sparse_marker, velocity, DT)
    velocity = advect.semi_lagrangian(velocity, velocity, DT)
