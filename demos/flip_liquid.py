""" FLIP simulation for liquids

A liquid block collides with a rotated obstacle and falls into a liquid pool.
"""

from phi.flow import *


DOMAIN = Domain(x=64, y=64, boundaries=CLOSED, bounds=Box[0:64, 0:64])
GRAVITY = math.tensor([0, -9.81])
DT = 0.1
OBSTACLE = Box[20:35, 30:35].rotated(-20)
ACCESSIBLE_MASK = DOMAIN.accessible_mask(OBSTACLE, type=StaggeredGrid)
_OBSTACLE_POINTS = DOMAIN.distribute_points(OBSTACLE, color='#000000')  # only for plotting

particles = DOMAIN.distribute_points(union(Box[20:40, 50:60], Box[:, :5])) * (0, 0)
velocity = particles >> DOMAIN.staggered_grid()
pressure = DOMAIN.scalar_grid()
scene = particles & _OBSTACLE_POINTS * (0, 0)  # only for plotting

for _ in ModuleViewer(display='scene').range():
    div_free_velocity, pressure, _, _, occupied = flip.make_incompressible(velocity + DT * GRAVITY, DOMAIN, ACCESSIBLE_MASK, particles)
    particles = flip.map_velocity_to_particles(particles, div_free_velocity, occupied, previous_velocity_grid=velocity)
    particles = advect.runge_kutta_4(particles, div_free_velocity, DT, accessible=ACCESSIBLE_MASK, occupied=occupied)
    particles = flip.respect_boundaries(particles, DOMAIN, [OBSTACLE])
    velocity = particles >> DOMAIN.staggered_grid()
    scene = particles & _OBSTACLE_POINTS * (0, 0)
