from unittest import TestCase

from phi.flow import *
from phi.physics.physics import STATIC


class CustomPhys(Physics):

    def __init__(self, name, list, deps):
        Physics.__init__(self, deps)
        self.list = list
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
        np.testing.assert_equal(order, ['Fan', 'Inflow'])

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
        except:
            pass
