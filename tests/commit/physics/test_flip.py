from unittest import TestCase

from phi.field._field_math import data_bounds
from phi.field._point_cloud import distribute_points
from phi.flow import *


def step(particles: PointCloud, obstacles: list, dt: float, bounds: Box, **grid_resolution):
    # --- Grid Operations ---
    velocity = prev_velocity = field.finite_fill(resample(particles, StaggeredGrid(0, 0, bounds, **grid_resolution), outside_handling='clamp', scatter=True))
    occupied = resample(field.mask(particles), CenteredGrid(0, velocity.extrapolation.spatial_gradient(), velocity.bounds, velocity.resolution), scatter=True)
    velocity, pressure = fluid.make_incompressible(velocity + (0, -9.81 * dt), obstacles, active=occupied)
    # --- Particle Operations ---
    particles += resample(velocity - prev_velocity, particles)  # FLIP update
    # particles = velocity @ particles  # PIC update
    particles = advect.points(particles, velocity * field.mask(~union(obstacles)), dt, advect.finite_rk4)
    particles = fluid.boundary_push(particles, obstacles + [~bounds])
    return particles


class FlipTest(TestCase):

    def test_falling_block_short(self):
        """ Tests if a block of liquid has a constant shape during free fall for 4 steps. """
        particles = initial_particles = distribute_points(union(Box['x,y', 12:20, 110:120]), x=32, y=128) * (0, -10)
        initial_bounds = data_bounds(particles)
        for i in range(4):
            particles = step(particles, [], dt=0.05, x=32, y=128, bounds=Box(x=32, y=128))
            math.assert_close(data_bounds(particles).size, initial_bounds.size)  # shape of falling block stays the same
            assert math.max(particles.center, dim='points').vector['y'] < math.max(initial_particles.points, dim='points').vector['y']  # block really falls

    def test_boundary_push(self):
        """ Tests if particles really get puhsed outside of obstacles and domain boundaries. """
        DOMAIN = Box(x=64, y=64)
        OBSTACLE = Box['x,y', 20:40, 10:30]
        particles = p0 = distribute_points(union(Box['x,y', 20:38, 20:50], Box['x,y', 50:60, 10:50]), center=True, x=64, y=64) * (10, 0)
        particles = pa = advect.points(particles, particles, 1)
        assert math.any(OBSTACLE.lies_inside(particles.points))
        assert math.any((~particles.bounds).lies_inside(particles.points))
        particles = pp = fluid.boundary_push(particles, [OBSTACLE, ~DOMAIN], separation=0.5)
        # show(expand(OBSTACLE, batch(list=3)), [p0['x'], pa['x'], pp['x']], overlay='args')
        assert math.all(~OBSTACLE.lies_inside(particles.points))
        assert math.all(DOMAIN.lies_inside(particles.center))
