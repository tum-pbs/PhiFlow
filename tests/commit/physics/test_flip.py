from unittest import TestCase

from phi.field._field_math import data_bounds
from phi.field._point_cloud import distribute_points
from phi.flow import *


def step(particles: PointCloud, accessible: StaggeredGrid, dt: float):
    velocity = particles @ accessible
    div_free_velocity, _, occupied = flip.make_incompressible(velocity + dt * math.tensor([0, -9.81]), particles, accessible)
    particles = flip.map_velocity_to_particles(particles, div_free_velocity, occupied, previous_velocity_grid=velocity, viscosity=0.9)
    particles = advect.runge_kutta_4(particles, div_free_velocity, dt, accessible=accessible, occupied=occupied)
    particles = flip.respect_boundaries(particles, [])
    return particles


class FlipTest(TestCase):

    def test_falling_block_short(self):
        """ Tests if a block of liquid has a constant shape during free fall for 4 steps. """
        ACCESSIBLE_FACES = field.stagger(CenteredGrid(1, 0, x=32, y=128), math.minimum, extrapolation.ZERO)
        particles = initial_particles = distribute_points(union(Box[12:20, 110:120]), x=32, y=128) * (0, -10)
        initial_bounds = data_bounds(particles)
        for i in range(4):
            particles = step(particles, ACCESSIBLE_FACES, dt=0.05)
            math.assert_close(data_bounds(particles).size, initial_bounds.size)  # shape of falling block stays the same
            assert math.max(particles.points, dim='points').vector['y'] < math.max(initial_particles.points, dim='points').vector['y']  # block really falls

    def test_respect_boundaries(self):
        """ Tests if particles really get puhsed outside of obstacles and domain boundaries. """
        OBSTACLE = Box[20:40, 10:30]
        particles = distribute_points(union(Box[20:38, 20:50], Box[50:60, 10:50]), center=True, x=64, y=64) * (10, 0)
        particles = advect.points(particles, particles, 1)
        assert math.any(OBSTACLE.lies_inside(particles.points))
        assert math.any((~particles.bounds).lies_inside(particles.points))
        particles = flip.respect_boundaries(particles, [OBSTACLE], offset=0.1)
        assert math.all(~OBSTACLE.lies_inside(particles.points))
        assert math.all(~(~particles.bounds).lies_inside(particles.points))


