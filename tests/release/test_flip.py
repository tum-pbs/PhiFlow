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

    def test_single_particles(self):
        """ Tests if single particles at the boundaries and within the domain really fall down. """
        ACCESSIBLE_FACES = field.stagger(CenteredGrid(1, 0, x=32, y=32), math.minimum, 0)
        particles = initial_particles = distribute_points(union(Box[0:1, 10:11], Box[31:32, 20:21], Box[10:11, 10:11]), x=32, y=32, points_per_cell=1) * (0, 0)
        self.assertEqual(3, particles.points.points.size)
        for i in range(10):
            particles = step(particles, ACCESSIBLE_FACES, dt=0.05)
            assert math.all(particles.points.vector[1] < initial_particles.points.vector[1])

    def test_pool(self):
        """ Tests if a pool of liquid at the bottom stays constant over time. """
        ACCESSIBLE_FACES = field.stagger(CenteredGrid(1, 0, x=32, y=32), math.minimum, 0)
        particles = initial_particles = distribute_points(Box[:, :10], x=32, y=32) * (0, 0)
        for i in range(100):
            particles = step(particles, ACCESSIBLE_FACES, dt=0.05)
        occupied_start = initial_particles.with_values(1) @ CenteredGrid(0, 0, x=32, y=32)
        occupied_end = particles.with_values(1) @ CenteredGrid(0, 0, x=32, y=32)
        math.assert_close(occupied_start.values, occupied_end.values)
        math.assert_close(initial_particles.points, particles.points, abs_tolerance=1e-3)

    def test_falling_block_long(self):
        """ Tests if a block of liquid has a constant shape during free fall. """
        ACCESSIBLE_FACES = field.stagger(CenteredGrid(1, 0, x=32, y=128), math.minimum, 0)
        particles = initial_particles = distribute_points(Box[12:20, 110:120], x=32, y=128) * (0, 0)
        initial_bounds = data_bounds(particles)
        for i in range(90):
            particles = step(particles, ACCESSIBLE_FACES, dt=0.05)
            math.assert_close(data_bounds(particles).size, initial_bounds.size)  # shape of falling block stays the same
            assert math.max(particles.points, dim='points').vector['y'] < math.max(initial_particles.points, dim='points').vector['y']  # block really falls

    def test_block_and_pool(self):
        """ Tests if the impact of a block on a pool has no side-effects (e.g. liquid explosion). """
        ACCESSIBLE_FACES = field.stagger(CenteredGrid(1, 0, x=32, y=32), math.minimum, 0)
        particles = distribute_points(union(Box[:, :5], Box[12:18, 15:20]), x=32, y=32) * (0, 0)
        for i in range(100):
            particles = step(particles, ACCESSIBLE_FACES, dt=0.05)
        assert math.all(particles.points.vector[1] < 15)

    def test_symmetry(self):
        """ Tests the symmetry of a setup where a liquid block collides with 2 rotated obstacles. """
        OBSTACLE = union([Box[20:30, 10:12].rotated(math.tensor(20)), Box[34:44, 10:12].rotated(math.tensor(-20))])
        ACCESSIBLE_FACES = field.stagger(CenteredGrid(~OBSTACLE, 0, x=64, y=64), math.minimum, 0)
        x_low = 26
        x_high = 38
        y_low = 40
        y_high = 50
        particles = distribute_points(Box[x_low:x_high, y_low:y_high], x=64, y=64, center=True) * (0, 0)
        x_num = int((x_high - x_low) / 2)
        y_num = y_high - y_low
        particles_per_cell = 8
        total = x_num * y_num
        for i in range(100):
            particles = step(particles, ACCESSIBLE_FACES, dt=0.05)
            left = particles.points.points[particles.points.vector[0] < 32]
            right = particles.points.points[particles.points.vector[0] > 32]
            self.assertEqual(left.points.size, right.points.size)
            mirrored = math.copy(right).numpy('points,vector')
            mirrored[:, 0] = 64 - right[:, 0]
            smirrored = np.zeros_like(mirrored)
            # --- particle order of mirrored version differs from original one and must be fixed for MSE
            # (caused by ordering in phi.physics._boundaries _distribute_points) ---
            for p in range(particles_per_cell):
                for b in range(x_num):
                    smirrored[p * total + b * y_num:p * total + (b + 1) * y_num] = mirrored[(p + 1) * total - (b + 1) * y_num:(p + 1) * total - b * y_num]
            mse = np.square(smirrored - left.numpy('points,vector')).mean()
            if i < 45:
                assert mse == 0  # block is still falling, hits obstacles at step 46
            else:
                assert mse <= 1e-3  # error increases gradually after block and obstacles collide
