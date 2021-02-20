from unittest import TestCase

from phi.flow import *


def step(particles, domain, dt, accessible):
    velocity = particles >> domain.staggered_grid()
    div_free_velocity, pressure, _, _, occupied = \
        flip.make_incompressible(velocity + dt * math.tensor([0, -9.81]), domain, accessible, particles)
    particles = flip.map_velocity_to_particles(particles, div_free_velocity, occupied, previous_velocity_grid=velocity)
    particles = advect.runge_kutta_4(particles, div_free_velocity, dt, accessible=accessible, occupied=occupied)
    particles = flip.respect_boundaries(particles, domain, [])
    return dict(particles=particles, domain=domain, dt=dt, accessible=accessible)


class FluidTest(TestCase):

    def test_single_particles(self):
        """ Tests if single particles at the boundaries and within the domain really fall down. """
        SIZE = 32
        DOMAIN = Domain(x=SIZE, y=SIZE, boundaries=CLOSED, bounds=Box[0:SIZE, 0:SIZE])
        DT = 0.05
        ACCESSIBLE = DOMAIN.accessible_mask([], type=StaggeredGrid)
        PARTICLES = DOMAIN.distribute_points(union(Box[0:1, 10:11], Box[31:32, 20:21], Box[10:11, 10:11]),
                                             points_per_cell=1) * (0, 0)
        assert PARTICLES.points.shape[0] == 3
        state = dict(particles=PARTICLES, domain=DOMAIN, dt=DT, accessible=ACCESSIBLE)
        for i in range(10):
            state = step(**state)
            assert math.all(state['particles'].points.native()[:, 1] < PARTICLES.points.native()[:, 1])
            PARTICLES = state['particles']

    def test_pool(self):
        """ Tests if a pool of liquid at the bottom stays constant over time. """
        SIZE = 32
        DOMAIN = Domain(x=SIZE, y=SIZE, boundaries=CLOSED, bounds=Box[0:SIZE, 0:SIZE])
        DT = 0.05
        ACCESSIBLE = DOMAIN.accessible_mask([], type=StaggeredGrid)
        PARTICLES = DOMAIN.distribute_points(union(Box[:, :10])) * (0, 0)

        state = dict(particles=PARTICLES, domain=DOMAIN, dt=DT, accessible=ACCESSIBLE)
        for i in range(100):
            state = step(**state)

        occupied_start = PARTICLES.with_(values=1) >> DOMAIN.grid()
        occupied_end = state['particles'].with_(values=1) >> DOMAIN.grid()
        math.assert_close(occupied_start.values, occupied_end.values)
        math.assert_close(PARTICLES.points, state['particles'].points, abs_tolerance=1e-5)

    def test_falling_block(self):
        """ Tests if a block of liquid has a constant shape during free fall. """
        DOMAIN = Domain(x=32, y=128, boundaries=CLOSED, bounds=Box[0:32, 0:128])
        DT = 0.05
        ACCESSIBLE = DOMAIN.accessible_mask([], type=StaggeredGrid)
        PARTICLES = DOMAIN.distribute_points(union(Box[12:20, 110:120]), center=True) * (0, 0)
        extent = math.max(PARTICLES.points, dim='points') - math.min(PARTICLES.points, dim='points')
        state = dict(particles=PARTICLES, domain=DOMAIN, dt=DT, accessible=ACCESSIBLE)
        for i in range(90):
            state = step(**state)
            curr_extent = math.max(state['particles'].points, dim='points') - \
                          math.min(state['particles'].points, dim='points')
            math.assert_close(curr_extent, extent)  # shape of falling block stays the same
            assert math.max(state['particles'].points, dim='points')[1] < \
                   math.max(PARTICLES.points, dim='points')[1]  # block really falls
            extent = curr_extent

    def test_block_and_pool(self):
        """ Tests if the impact of a block on a pool has no sideeffects (e.g. liquid explosion). """
        SIZE = 32
        DOMAIN = Domain(x=SIZE, y=SIZE, boundaries=CLOSED, bounds=Box[0:SIZE, 0:SIZE])
        DT = 0.05
        ACCESSIBLE = DOMAIN.accessible_mask([], type=StaggeredGrid)
        PARTICLES = DOMAIN.distribute_points(union(Box[:, :5], Box[12:18, 15:20])) * (0, 0)

        state = dict(particles=PARTICLES, domain=DOMAIN, dt=DT, accessible=ACCESSIBLE)
        for i in range(100):
            state = step(**state)

        assert math.all(state['particles'].points.native()[:, 1] < 15)

    def test_respect_boundaries(self):
        """ Tests if particles really get puhsed outside of obstacles and domain boundaries. """
        SIZE = 64
        DOMAIN = Domain(x=SIZE, y=SIZE, boundaries=CLOSED, bounds=Box[0:SIZE, 0:SIZE])
        OBSTACLE = Box[20:40, 10:30]
        PARTICLES = DOMAIN.distribute_points(union(Box[20:38, 20:50], Box[50:60, 10:50]), center=True) * (10, 0)
        PARTICLES = advect.points(PARTICLES, PARTICLES, 1)
        assert math.any(OBSTACLE.lies_inside(PARTICLES.points))
        assert math.any((~DOMAIN.bounds).lies_inside(PARTICLES.points))
        PARTICLES = flip.respect_boundaries(PARTICLES, DOMAIN, [OBSTACLE], offset=0.1)
        assert math.all(~OBSTACLE.lies_inside(PARTICLES.points))
        assert math.all(~(~DOMAIN.bounds).lies_inside(PARTICLES.points))

    def test_symmetry(self):
        """ Tests the symmetry of a setup where a liquid block collides with 2 rotated obstacles. """
        SIZE = 64
        MID = SIZE / 2
        DOMAIN = Domain(x=SIZE, y=SIZE, boundaries=CLOSED, bounds=Box[0:SIZE, 0:SIZE])
        DT = 0.05
        OBSTACLE = union([Box[20:30, 10:12].rotated(math.tensor(20)), Box[34:44, 10:12].rotated(math.tensor(-20))])
        ACCESSIBLE = DOMAIN.accessible_mask(OBSTACLE, type=StaggeredGrid)
        x_low = 26
        x_high = 38
        y_low = 40
        y_high = 50
        PARTICLES = DOMAIN.distribute_points(union(Box[x_low:x_high, y_low:y_high]), center=True) * (0, 0)

        x_num = int((x_high - x_low) / 2)
        y_num = y_high - y_low
        particles_per_cell = 8
        total = x_num * y_num

        state = dict(particles=PARTICLES, domain=DOMAIN, dt=DT, accessible=ACCESSIBLE)
        for i in range(100):
            state = step(**state)
            particles = state['particles'].points.numpy()
            left = particles[particles[:, 0] < MID]
            right = particles[particles[:, 0] > MID]
            assert len(left) == len(right)
            mirrored = right.copy()
            mirrored[:, 0] = 2 * MID - right[:, 0]
            smirrored = np.zeros_like(mirrored)
            # --- particle order of mirrored version differs from original one and
            # must be fixed for MSE ---
            for p in range(particles_per_cell):
                for b in range(x_num):
                    smirrored[p * total + b * y_num:p * total + (b + 1) * y_num] = \
                        mirrored[(p + 1) * total - (b + 1) * y_num:(p + 1) * total - b * y_num]
            mse = np.square(smirrored - left).mean()
            if i < 45:
                assert mse == 0  # block was falling until this step, hits obstacles at step 46
            elif i < 80:
                assert mse <= 1e-7  # error increases gradually after block and obstacles collide
            else:
                assert mse <= 1e-3

