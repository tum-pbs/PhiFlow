from unittest import TestCase

from phi import field
from phi.field import Noise, CenteredGrid, StaggeredGrid
from phi.field._point_cloud import distribute_points
from phi.physics import advect


def _test_advection(adv):
        s = CenteredGrid(Noise(), x=4, y=3)
        v = CenteredGrid(Noise(vector=2), x=4, y=3)
        field.assert_close(s, adv(s, v, 0), adv(s, v * 0, 1))
        sv = StaggeredGrid(Noise(), x=4, y=3)
        field.assert_close(s, adv(s, sv, 0), adv(s, sv * 0, 1))
        field.assert_close(sv, adv(sv, sv, 0), adv(sv, sv * 0, 1))


class TestAdvect(TestCase):

    def test_advect(self):
        _test_advection(advect.advect)

    def test_semi_lagrangian(self):
        _test_advection(advect.semi_lagrangian)

    def test_mac_cormack(self):
        _test_advection(advect.mac_cormack)

    def test_advect_points_euler(self):
        v = distribute_points(1, points_per_cell=2, x=4, y=3) * (1, -1)
        field.assert_close(v, advect.points(v, v, 0), advect.points(v, v*0, 0))

    def test_advect_points_rk4(self):
        v = distribute_points(1, points_per_cell=2, x=4, y=3) * (1, -1)
        field.assert_close(v, advect.points(v, v, 0, advect.rk4), advect.points(v, v*0, 0, advect.rk4))
        field.assert_close(v, advect.points(v, v, 0, advect.finite_rk4), advect.points(v, v*0, 0, advect.finite_rk4))
