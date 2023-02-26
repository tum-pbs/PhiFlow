""" Point Cloud Demo
Demonstrates working with PointCloud objects and plotting them.
"""
from phi.flow import *


points1 = PointCloud(vec(x=1, y=1))
points2 = PointCloud(vec(x=20, y=20))
# points = points1 & points2
points = field.stack([points1, points2], instance('points'))

# Advection
velocity = CenteredGrid((-1, 1), x=64, y=64, bounds=Box(x=100, y=100))
points = advect.advect(points, velocity, 10)  # RK4
points = advect.advect(points, points * (-1, 1), -5)  # Euler

# Grid sampling
scattered_data = field.sample(points, velocity.elements, scatter=True)
scattered_grid = points.at(velocity, scatter=True)
scattered_sgrid = resample(points, to=StaggeredGrid(0, 0, velocity.bounds, velocity.resolution), scatter=True)

view(namespace=globals())
