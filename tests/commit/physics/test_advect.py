from unittest import TestCase

from phi import field
from phi.field import Noise, CenteredGrid, StaggeredGrid
from phi.field._point_cloud import distribute_points
from phi.geom import Box
from phi.physics import advect
from phiml import math
from phiml.math import spatial, wrap


def _test_advection(adv):
        s = CenteredGrid(Noise(), x=4, y=3)
        v = CenteredGrid(Noise(vector='x,y'), x=4, y=3)
        field.assert_close(s, adv(s, v, 0), adv(s, v * 0, 1), abs_tolerance=1e-5)
        sv = StaggeredGrid(Noise(), x=4, y=3)
        field.assert_close(s, adv(s, sv, 0), adv(s, sv * 0, 1), abs_tolerance=1e-5)
        field.assert_close(sv, adv(sv, sv, 0), adv(sv, sv * 0, 1), abs_tolerance=1e-5)


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

    def test_self_advect_staggered(self):
        v0 = StaggeredGrid(Box(x=(.9, 2.6), y=(.9, 2)), 0, x=4, y=3) * (0, 1)
        v = advect.semi_lagrangian(v0, v0, 1)
        math.assert_close(0, v['x'].values)
        math.assert_close(wrap([[0, 0, 0, 0], [0, 1, 1, 0]], spatial('y,x')), v['y'].values)
