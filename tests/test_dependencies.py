from unittest import TestCase

import numpy

from phi.geom import box
from phi.physics.field.effect import Inflow, Fan
from phi.physics.physics import STATIC, Physics, StateDependency
from phi.physics.world import World


class CustomPhys(Physics):

    def __init__(self, name, order_list, deps):
        Physics.__init__(self, deps)
        self.list = order_list
        self.name = name

    def step(self, state, dt=1.0, **dependent_states):
        self.list.append(self.name)
        return STATIC.step(state, dt, **dependent_states)


class TestDependencies(TestCase):

    def test_order(self):
        world = World()
        order = []
        world.add(Inflow(box[0:0]), physics=CustomPhys('Inflow', order, [StateDependency('d', 'fan', blocking=True)]))
        world.add(Fan(box[0:0], 0), physics=CustomPhys('Fan', order, []))
        world.step()
        numpy.testing.assert_equal(order, ['Fan', 'Inflow'])

    def test_cyclic_dependency(self):
        world = World()
        order = []
        inflow = world.add(Inflow(box[0:0]))
        inflow.physics = CustomPhys('Inflow', order, [StateDependency('d', 'fan', blocking=True)])
        fan = world.add(Fan(box[0:0], 0))
        fan.physics = CustomPhys('Fan', order, [StateDependency('d', 'inflow', blocking=True)])
        try:
            world.step()
            self.fail('Cycle not recognized.')
        except AssertionError:
            pass
