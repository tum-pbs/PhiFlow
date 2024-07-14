""" Streamline Profile
Simulates a viscous fluid flowing through a horizontal pipe.
"""
from unittest import TestCase

from phiml import math
from phiml.math import extrapolation, tensor, channel

from phi.jax import JAX
from phi.geom import Box
from phi.physics import fluid, advect, diffuse
from phi import field
from phi.field import StaggeredGrid, CenteredGrid


class Higher_Order_test(TestCase):

    def _test_higher_order(self, order, xynum, vis, dt, jit_compile, t_num, freq, boundary_vel, at):

        def fourth_ord_runge_kutta(velocity, pressure):

            def adv_diff_press(v, p):
                adv_diff_press = (advect.finite_difference(v, v, order=order))
                diff = (diffuse.finite_difference(v, vis, order=order))
                adv_diff_press += diff
                adv_diff_press -= field.spatial_gradient(p, at=at, order=order, gradient_extrapolation=extrapolation.ZERO)
                return adv_diff_press.with_extrapolation(0)

            def pressure_treatment(v, p, dt_):
                v, delta_p = \
                    fluid.make_incompressible(v, solve=math.Solve('biCG-stab(2)', 1e-10, 1e-10), order=order)
                p += delta_p / dt_
                return v, p

            v_1, p_1 = velocity, pressure
            rhs_1 = adv_diff_press(v_1, p_1)
            v_2_old = velocity + (dt / 2) * rhs_1
            v_2, p_2 = pressure_treatment(v_2_old, p_1, dt / 2)

            rhs_2 = adv_diff_press(v_2, p_2)
            v_3_old = velocity + (dt / 2) * rhs_2
            v_3, p_3 = pressure_treatment(v_3_old, p_2, dt / 2)

            rhs_3 = adv_diff_press(v_3, p_3)
            v_4_old = velocity + dt * rhs_2
            v_4, p_4 = pressure_treatment(v_4_old, p_3, dt)

            rhs_4 = adv_diff_press(v_4, p_4)
            v_p1_old = velocity + (dt / 6) * (rhs_1 + 2 * rhs_2 + 2 * rhs_3 + rhs_4)
            p_p1_old = (1 / 6) * (p_1 + 2 * p_2 + 2 * p_3 + p_4)
            v_p1, p_p1 = pressure_treatment(v_p1_old, p_p1_old, dt)

            return v_p1, p_p1

        if jit_compile:
            timestepper = math.jit_compile(fourth_ord_runge_kutta)
        else:
            timestepper = fourth_ord_runge_kutta

        with JAX:
            with math.precision(64):
                DOMAIN_V = dict(bounds=Box['x,y', 0:1, 0:1], x=xynum, y=xynum, extrapolation=extrapolation.combine_sides(
                                  x=extrapolation.ZERO,
                                  y=(extrapolation.ZERO, extrapolation.combine_by_direction(extrapolation.ZERO, extrapolation.ConstantExtrapolation(boundary_vel)))))
                DOMAIN_P = dict(bounds=Box['x,y', 0:1, 0:1], x=xynum, y=xynum, extrapolation=extrapolation.ZERO_GRADIENT)
                if at == 'face':
                    velocity = StaggeredGrid(tensor([0, 0], channel(vector='x, y')), **DOMAIN_V)
                elif at == 'center':
                    velocity = CenteredGrid(tensor([0, 0], channel(vector='x, y')), **DOMAIN_V)
                pressure = CenteredGrid(0, **DOMAIN_P)
                for i in range(1, t_num+1):
                    if i % freq == 0:
                        print(f"timestep: {i} of {t_num}")
                        velocity, pressure = timestepper(velocity, pressure)
                return velocity, pressure

    def test_higher_order(self):
        self._test_higher_order(2, 31, 1/1000, 0.001, False, 2, 1, -1, 'face')
        self._test_higher_order(6, 31, 1 / 1000, 0.001, False, 2, 1, -1, 'center')

    def test_higher_order_jit(self):
        self._test_higher_order(2, 31, 1/1000, 0.001, True, 100, 1, -1, 'face')
        self._test_higher_order(6, 31, 1 / 1000, 0.001, True, 100, 1, -1, 'center')

    def test_higher_order_values(self):
        v, p = self._test_higher_order(2, 31, 1/1000, 0.001, True, 100, 10, 0, 'face')
        math.assert_close(0, v.values, abs_tolerance=2e-5)
        math.assert_close(0, p.values, abs_tolerance=2e-5)
        v, p = self._test_higher_order(6, 31, 1 / 1000, 0.001, True, 100, 10, 0, 'center')
        math.assert_close(0, v.values, abs_tolerance=2e-5)
        math.assert_close(0, p.values, abs_tolerance=2e-5)
