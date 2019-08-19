from unittest import TestCase
from phi.flow import *


class CustomPhysics(Physics):

    def __init__(self, name, list, deps, blocking):
        Physics.__init__(self, deps, blocking)
        self.list = list
        self.name = name

    def step(self, state, dt=1.0, **dependent_states):
        self.list.append(self.name)
        return STATIC.step(state, dt, **dependent_states)


class TestDependencies(TestCase):

    def test_order(self):
        world = World()
        order = []
        world.Inflow(box[0:0], physics=CustomPhysics('Inflow', order, {}, {'d': 'fan'}))
        world.Fan(box[0:0], 0, physics=CustomPhysics('Fan', order, {}, {}))
        world.step()
        np.testing.assert_equal(order, ['Fan', 'Inflow'])

    def test_cyclic_dependency(self):
        world = World()
        order = []
        inflow = world.Inflow(box[0:0])
        inflow.physics = CustomPhysics('Inflow', order, {}, {'d': 'fan'})
        fan = world.Fan(box[0:0], 0)
        fan.physics = CustomPhysics('Fan', order, {}, {'d': 'inflow'})
        try:
            world.step()
            self.fail('Cycle not recognized.')
        except:
            pass