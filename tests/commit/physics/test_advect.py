from unittest import TestCase

from phi import field
from phi.field import Noise
from phi.physics import advect
from phi.physics._boundaries import Domain


def _test_advection(adv):
        domain = Domain(x=4, y=3)
        s = domain.scalar_grid(Noise())
        v = domain.vector_grid(Noise(vector=2))
        field.assert_close(s, adv(s, v, 0), adv(s, v * 0, 1))
        sv = domain.staggered_grid(Noise())
        field.assert_close(s, adv(s, sv, 0), adv(s, sv * 0, 1))
        field.assert_close(sv, adv(sv, sv, 0), adv(sv, sv * 0, 1))


class TestAdvect(TestCase):

    def test_advect(self):
        _test_advection(advect.advect)

    def test_semi_lagrangian(self):
        _test_advection(advect.semi_lagrangian)

    def test_mac_cormack(self):
        _test_advection(advect.mac_cormack)

    def test_advect_points(self):
        domain = Domain(x=4, y=3)
        v = domain.distribute_points(domain.bounds, points_per_cell=2) * (1, -1)
        field.assert_close(v, advect.points(v, v, 0), advect.points(v, v*0, 0))

    def test_runge_kutta_4(self):
        domain = Domain(x=4, y=3)
        points = domain.distribute_points(domain.bounds, points_per_cell=2)
        v = domain.vector_grid(Noise(vector=2))
        field.assert_close(points, advect.runge_kutta_4(points, v, 0), advect.runge_kutta_4(points, v*0, 0))
        sv = domain.staggered_grid(Noise())
        field.assert_close(points, advect.runge_kutta_4(points, sv, 0), advect.runge_kutta_4(points, sv*0, 0))
