""" Rotating Bar
This demo shows how to simulate fluid flow with moving or rotating obstacles.
"""
from phi.flow import *


DOMAIN = Domain(x=100, y=100, boundaries=OPEN, bounds=Box[0:100, 0:100])
DT = 1.0
obstacle = Obstacle(Box[47:53, 20:70], angular_velocity=0.05)
obstacle_mask = DOMAIN.scalar_grid(obstacle.geometry)  # to show in user interface
velocity = DOMAIN.staggered_grid((1, 0))

for frame in ModuleViewer(framerate=10, display=('velocity', 'obstacle_mask'), autorun=True).range():
    obstacle = obstacle.copied_with(geometry=obstacle.geometry.rotated(-obstacle.angular_velocity * DT))  # rotate bar
    velocity = advect.mac_cormack(velocity, velocity, DT)
    velocity, pressure, _iter, _ = fluid.make_incompressible(velocity, DOMAIN, (obstacle,), solve_params=math.LinearSolve(absolute_tolerance=1e-2, max_iterations=1e5))
    print(f"{frame}: {_iter}")
    obstacle_mask = DOMAIN.scalar_grid(obstacle.geometry)
