from unittest import TestCase

from phi.flow import *
from phi.physics._boundaries import Domain, STICKY


def step(particles, domain, dt, accessible):
    velocity = particles @ domain.staggered_grid()
    div_free_velocity, _, occupied = \
        flip.make_incompressible(velocity + dt * math.tensor([0, -9.81]), domain, particles, accessible)
    particles = flip.map_velocity_to_particles(particles, div_free_velocity, occupied, previous_velocity_grid=velocity, viscosity=0.9)
    particles = advect.runge_kutta_4(particles, div_free_velocity, dt, accessible=accessible, occupied=occupied)
    particles = flip.respect_boundaries(particles, domain, [])
    return dict(particles=particles, domain=domain, dt=dt, accessible=accessible)


class FlipTest(TestCase):

    def test_falling_block_short(self):
        """ Tests if a block of liquid has a constant shape during free fall for 4 steps. """
        DOMAIN = Domain(x=32, y=128, boundaries=STICKY, bounds=Box[0:32, 0:128])
        DT = 0.05
        ACCESSIBLE = DOMAIN.accessible_mask([], type=StaggeredGrid)
        PARTICLES = DOMAIN.distribute_points(union(Box[12:20, 110:120])) * (0, -10)
        extent = math.max(PARTICLES.points, dim='points') - math.min(PARTICLES.points, dim='points')
        state = dict(particles=PARTICLES, domain=DOMAIN, dt=DT, accessible=ACCESSIBLE)
        for i in range(4):
            state = step(**state)
            curr_extent = math.max(state['particles'].points, dim='points') - \
                          math.min(state['particles'].points, dim='points')
            math.assert_close(curr_extent, extent)  # shape of falling block stays the same
            assert math.max(state['particles'].points, dim='points')[1] < \
                   math.max(PARTICLES.points, dim='points')[1]  # block really falls
            extent = curr_extent

    def test_respect_boundaries(self):
        """ Tests if particles really get puhsed outside of obstacles and domain boundaries. """
        SIZE = 64
        DOMAIN = Domain(x=SIZE, y=SIZE, boundaries=STICKY, bounds=Box[0:SIZE, 0:SIZE])
        OBSTACLE = Box[20:40, 10:30]
        PARTICLES = DOMAIN.distribute_points(union(Box[20:38, 20:50], Box[50:60, 10:50]), center=True) * (10, 0)
        PARTICLES = advect.points(PARTICLES, PARTICLES, 1)
        assert math.any(OBSTACLE.lies_inside(PARTICLES.points))
        assert math.any((~DOMAIN.bounds).lies_inside(PARTICLES.points))
        PARTICLES = flip.respect_boundaries(PARTICLES, DOMAIN, [OBSTACLE], offset=0.1)
        assert math.all(~OBSTACLE.lies_inside(PARTICLES.points))
        assert math.all(~(~DOMAIN.bounds).lies_inside(PARTICLES.points))


