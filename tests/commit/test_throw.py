from unittest import TestCase

import phi
from phi import math
from phi.field import CenteredGrid
from phi.geom import Box
from phi.math import channel, tensor
from phi.math.backend import Backend


def simulate_hit(pos, height, vel, angle, gravity=1.):
    vel_x, vel_y = math.cos(angle) * vel, math.sin(angle) * vel
    height = math.maximum(height, .5)
    hit_time = (vel_y + math.sqrt(vel_y**2 + 2 * gravity * height)) / gravity
    return pos + vel_x * hit_time, hit_time, height, vel_x, vel_y


def sample_trajectory(pos, height, vel, angle, gravity=1.):
    hit, hit_time, height, vel_x, vel_y = simulate_hit(pos, height, vel, angle, gravity)
    def y(x):
        t = (x.vector[0] - pos) / vel_x
        y_ = height + vel_y * t - gravity / 2 * t ** 2
        return math.where((y_ > 0) & (t > 0), y_, math.NAN)
    return CenteredGrid(y, x=2000, bounds=Box(x=(min(pos.min, hit.min), max(pos.max, hit.max))))


BACKENDS = phi.detect_backends()


class TestThrow(TestCase):

    def test_simulate_hit(self):
        math.assert_close(10 + math.sqrt(2), simulate_hit(10, 1, 1, 0)[0])

    def test_sample_trajectory(self):
        sample_trajectory(tensor(10), 1, 1, math.linspace(-math.PI / 4, 1.5, channel(linspace=7)))

    def test_gradient_descent(self):
        def loss_function(vel):
            return math.l2_loss(simulate_hit(10, 1, vel, 0)[0] - 0)
        gradient = math.functional_gradient(loss_function)

        for backend in BACKENDS:
            if backend.supports(Backend.jacobian):
                with backend:
                    vel = 1
                    for i in range(10):
                        loss, (grad,) = gradient(vel)
                        vel = vel - .2 * grad
                        print(f"vel={vel} - loss={loss}")
                    math.assert_close(-7.022265, vel)
