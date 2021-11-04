""" Point Cloud Demo
Demonstrates working with PointCloud objects and plotting them.
"""

from phi.physics._boundaries import Domain, STICKY as CLOSED
from phi.flow import *


DOMAIN = Domain(x=64, y=64, boundaries=CLOSED, bounds=Box(0, (100, 100)))
# points = DOMAIN.points([(1, 1), (20, 20)], color=['#ba0a04', '#344feb'])

points1 = DOMAIN.points((1, 1), color='#ba0a04')
points2 = DOMAIN.points((20, 20), color='#344feb')
# points = points1 & points2
points = field.concat([points1, points2], instance('points'))

# Advection
velocity = DOMAIN.vector_grid([-1, 1])
points = advect.advect(points, velocity, 10)  # RK4
points = advect.advect(points, points * (-1, 1), -5)  # Euler

# Grid sampling
scattered_data = field.sample(points, DOMAIN.cells)
scattered_grid = points @ DOMAIN.vector_grid()
scattered_sgrid = points @ DOMAIN.staggered_grid()

view(namespace=globals())
