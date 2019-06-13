from unittest import TestCase
from phi.flow import *


class TestSmoke(TestCase):
    def test_simple_smoke(self):
        smoke = Smoke(Domain([16, 16]))
        state = smoke.initial_state()
        smoke.step(state)

    def test_simpleplume(self):
        world.reset()
        smoke = world.Smoke(Domain([16, 16]))
        inflow = world.Inflow(Sphere((8, 8), radius=4))
        state = smoke.initial_state()
        state = state * smoke.step * smoke.step
