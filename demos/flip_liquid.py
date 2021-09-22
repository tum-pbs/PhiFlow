""" FLIP simulation for liquids

A liquid block collides with a rotated obstacle and falls into a liquid pool.
"""

from phi.physics._boundaries import Domain, STICKY as CLOSED
from phi.flow import *
# from phi.torch.flow import *
# from phi.tf.flow import *
# from phi.jax.flow import *


DOMAIN = Domain(x=64, y=64, boundaries=CLOSED, bounds=Box[0:64, 0:64])
GRAVITY = math.tensor([0, -9.81])
DT = 0.1
OBSTACLE = Box[1:35, 30:33].rotated(-20)
ACCESSIBLE_MASK = field.stagger(CenteredGrid(~OBSTACLE, extrapolation.ZERO, x=64, y=64), math.minimum, extrapolation.ZERO)
_OBSTACLE_POINTS = DOMAIN.distribute_points(OBSTACLE, color='#000000', points_per_cell=1, center=True)  # only for plotting

particles = DOMAIN.distribute_points(union(Box[15:30, 50:60], Box[:, :5])) * (0, 0)
# particles = nonzero(CenteredGrid(union(Box[15:30, 50:60], Box[:, :5]), 0, **DOMAIN)) * (0, 0)
velocity = particles @ DOMAIN.staggered_grid()
pressure = DOMAIN.scalar_grid()
scene = particles & _OBSTACLE_POINTS * (0, 0)  # only for plotting

for _ in view('scene,velocity,pressure', display='scene', play=False, namespace=globals()).range():
    div_free_velocity, _, occupied = flip.make_incompressible(velocity + DT * GRAVITY, DOMAIN, particles, ACCESSIBLE_MASK)
    particles = flip.map_velocity_to_particles(particles, div_free_velocity, occupied, previous_velocity_grid=velocity)
    particles = advect.runge_kutta_4(particles, div_free_velocity, DT, accessible=ACCESSIBLE_MASK, occupied=occupied)
    particles = flip.respect_boundaries(particles, DOMAIN, [OBSTACLE])
    velocity = particles @ DOMAIN.staggered_grid()
    scene = particles & _OBSTACLE_POINTS * (0, 0)
