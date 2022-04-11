""" FLIP simulation for liquids

A liquid block collides with a rotated obstacle and falls into a liquid pool.
"""
from phi.field._point_cloud import distribute_points
from phi.flow import *
# from phi.torch.flow import *
# from phi.tf.flow import *
# from phi.jax.flow import *


GRAVITY = math.tensor([0, -9.81])
DT = 0.1
OBSTACLE = Box(x=(1, 25), y=(30, 33)).rotated(-20)
ACCESSIBLE_CELLS = CenteredGrid(~OBSTACLE, 0, x=64, y=64)
ACCESSIBLE_FACES = field.stagger(ACCESSIBLE_CELLS, math.minimum, extrapolation.ZERO)
_OBSTACLE_POINTS = PointCloud(Cuboid(field.support(1 - ACCESSIBLE_CELLS, 'points'), x=2, y=2), color='#000000', bounds=ACCESSIBLE_CELLS.bounds)

particles = distribute_points(union(Box(x=(15, 30), y=(50, 60)), Box(x=None, y=(-INF, 5))), x=64, y=64) * (0, 0)
velocity = particles @ StaggeredGrid(0, 0, x=64, y=64)
scene = vis.overlay(particles, _OBSTACLE_POINTS)  # only for plotting

for _ in view('scene,velocity', display='scene', play=False, namespace=globals()).range():
    div_free_velocity, _, occupied = flip.make_incompressible(velocity + DT * GRAVITY, particles, ACCESSIBLE_FACES)
    particles = flip.map_velocity_to_particles(particles, div_free_velocity, occupied, previous_velocity_grid=velocity)
    particles = advect.runge_kutta_4(particles, div_free_velocity, DT, accessible=ACCESSIBLE_FACES, occupied=occupied)
    particles = flip.respect_boundaries(particles, [OBSTACLE])
    velocity = particles @ velocity
    scene = vis.overlay(particles, _OBSTACLE_POINTS)
