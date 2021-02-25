""" Passive Markers

Fluid simulation with additional marker fields that are passively transported with the fluid.

The dense marker is sampled on a regular grid while the sparse marker is a collection of particles.
"""

from phi.flow import *


def checkerboard(size=8, offset=2):
    return math.to_float(math.all((DOMAIN.cells.center - offset) % (2 * size) < size, 'vector'))


DOMAIN = Domain(x=126, y=160, boundaries=CLOSED)
DT = 0.2

velocity = DOMAIN.staggered_grid(Noise(vector=2, scale=100)) * 4
dense_marker = CenteredGrid(checkerboard(), DOMAIN.bounds)
points = math.join_dimensions(DOMAIN.cells.center.x[::4].y[::4], ('x', 'y'), 'points').points.as_batch()
sparse_marker = PointCloud(Sphere(points, 1), math.random_normal(points.shape.without('vector')))

for _ in ModuleViewer(framerate=10).range():
    velocity = advect.semi_lagrangian(velocity, velocity, DT)
    velocity, _, _, _ = fluid.make_incompressible(velocity, DOMAIN)
    dense_marker = advect.advect(dense_marker, velocity, DT)
    sparse_marker = advect.advect(sparse_marker, velocity, DT)
