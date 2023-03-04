""" Passive Markers

Fluid simulation with additional marker fields that are passively transported with the fluid.

The dense marker is sampled on a regular grid while the sparse marker is a collection of particles.
"""

from phi.flow import *


DOMAIN = dict(x=64, y=64, bounds=Box(x=100, y=100))
DT = 0.2
INITIAL_LOC = math.meshgrid(x=8, y=8).pack('x,y', instance('points')) * 10. + 10.

velocity = StaggeredGrid(Noise(vector='x,y', scale=100), 0, **DOMAIN) * 4
sparse_marker = PointCloud(Sphere(INITIAL_LOC, 2), 1, 0, bounds=DOMAIN['bounds'])
dense_marker = CenteredGrid(sparse_marker.elements, ZERO_GRADIENT, x=200, y=200, bounds=DOMAIN['bounds'])

for _ in view(framerate=10, play=False, namespace=globals()).range():
    velocity, _ = fluid.make_incompressible(velocity)
    dense_marker = advect.advect(dense_marker, velocity, DT)
    sparse_marker = advect.advect(sparse_marker, velocity, DT)
    velocity = advect.semi_lagrangian(velocity, velocity, DT)
