""" Rotating Bar
This demo shows how to simulate fluid flow with moving or rotating obstacles.
"""
from phi.flow import *


DOMAIN = dict(x=100, y=100, bounds=Box[0:100, 0:100])
DT = 1.0
obstacle = Obstacle(Box[47:53, 20:70], angular_velocity=0.05)
obstacle_mask = CenteredGrid(obstacle.geometry, extrapolation.ZERO, **DOMAIN)  # to show in user interface
velocity = StaggeredGrid(0, extrapolation.ZERO, **DOMAIN)

for frame in view(velocity, obstacle_mask, namespace=globals(), framerate=10, display=('velocity', 'obstacle_mask')).range():
    obstacle = obstacle.copied_with(geometry=obstacle.geometry.rotated(-obstacle.angular_velocity * DT))  # rotate bar
    velocity = advect.mac_cormack(velocity, velocity, DT)
    velocity, pressure = fluid.make_incompressible(velocity, (obstacle,))
    fluid.masked_laplace.tracers.clear()  # we will need to retrace because the matrix changes each step. This is not needed when JIT-compiling the physics.
    obstacle_mask = CenteredGrid(obstacle.geometry, extrapolation.ZERO, **DOMAIN)
