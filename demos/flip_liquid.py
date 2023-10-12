""" FLIP simulation for liquids

A liquid block collides with a rotated obstacle and falls into a liquid pool.
"""
from tqdm import trange

from phi.field._point_cloud import distribute_points
from phi.flow import *
# from phi.torch.flow import *
# from phi.tf.flow import *
# from phi.jax.flow import *


DOMAIN = Box(x=64, y=64)
GRAVITY = vec(x=0, y=-9.81)
OBSTACLE = Box(x=(1, 25), y=(30, 33)).rotated(-20)
ACCESSIBLE_CELLS = CenteredGrid(~OBSTACLE, 0, x=64, y=64)
_OBSTACLE_POINTS = PointCloud(Cuboid(field.support(1 - ACCESSIBLE_CELLS, 'points'), x=2, y=2))

particles = distribute_points(union(Box(x=(15, 30), y=(50, 60)), Box(x=None, y=(-INF, 5))), x=64, y=64) * (0, 0)
scene = vis.overlay(particles, _OBSTACLE_POINTS)  # only for plotting


@jit_compile
def step(particles, dt=.2, velocity=None, pressure=None):
    # --- Grid Operations ---
    velocity = prev_velocity = field.finite_fill(resample(particles, StaggeredGrid(0, 0, x=64, y=64), scatter=True, outside_handling='clamp'))
    occupied = resample(field.mask(particles), CenteredGrid(0, velocity.extrapolation.spatial_gradient(), velocity.bounds, velocity.resolution), scatter=True, outside_handling='clamp')
    velocity, pressure = fluid.make_incompressible(velocity + GRAVITY * dt, [OBSTACLE], active=occupied)
    # --- Particle Operations ---
    particles += resample(velocity - prev_velocity, to=particles)  # FLIP update
    # particles = resample(velocity, particles)  # PIC update
    particles = advect.points(particles, velocity * mask(~OBSTACLE), dt, advect.finite_rk4)
    particles = fluid.boundary_push(particles, [OBSTACLE, ~DOMAIN])
    return particles, velocity, pressure


# trj = iterate(step, batch(t=100), particles, range=trange)
# plot(vis.overlay(particles.with_values(1), _OBSTACLE_POINTS), animate='t')

for i in trange(100):
    particles, velocity, pressure = step(particles)
    scene = vis.overlay(particles.with_values(1), _OBSTACLE_POINTS)
    if i % 5 == 0:
        vis.show(scene, pressure)
