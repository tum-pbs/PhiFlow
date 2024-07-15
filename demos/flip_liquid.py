""" FLIP simulation for liquids

A liquid block collides with a rotated obstacle and falls into a liquid pool.
"""
from tqdm import trange

from phi.field._point_cloud import distribute_points
from phi.flow import *
# from phi.torch.flow import *
# from phi.tf.flow import *
# from phi.jax.flow import *


domain = Box(x=64, y=64)
obstacle = Box(x=(1, 25), y=(30, 33)).rotated(-20)
particles = distribute_points(union(Box(x=(15, 30), y=(50, 60)), Box(x=None, y=(-INF, 5))), x=64, y=64) * (0, 0)


@jit_compile
def step(particles: Field, pressure=None, dt=.1, gravity=vec(x=0, y=-9.81)):
    # --- Grid Operations ---
    grid_v = prev_grid_v = field.finite_fill(particles.at(StaggeredGrid(0, 0, domain, x=64, y=64), scatter=True, outside_handling='clamp'))
    occupied = resample(field.mask(particles), CenteredGrid(0, grid_v.extrapolation.spatial_gradient(), grid_v.bounds, grid_v.resolution), scatter=True, outside_handling='clamp')
    grid_v, pressure = fluid.make_incompressible(grid_v + gravity * dt, [obstacle], active=occupied)
    # --- Particle Operations ---
    particles += resample(grid_v - prev_grid_v, to=particles)  # FLIP update
    # particles = resample(grid_v, particles)  # PIC update
    particles = advect.points(particles, grid_v * resample(~obstacle, to=grid_v), dt, advect.finite_rk4)
    particles = fluid.boundary_push(particles, [obstacle, ~domain], separation=.5)
    return particles, pressure


for i in trange(100):
    particles, pressure = step(particles)
    if i % 5 == 0:
        vis.show(resample(pressure, to=particles), obstacle, overlay='args')
