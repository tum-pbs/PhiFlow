from unittest import TestCase

import numpy

from phi.physics.collective import CollectiveState
from phi.physics.domain import Domain
from phi.physics.smoke import Smoke
from phi.physics.world import World


class TestWorld(TestCase):

    def test_names(self):
        c = CollectiveState()
        self.assertEqual(c.states, {})
        c = c.state_added(Smoke(Domain([64])))
        try:
            c = c.state_added(Smoke(Domain([80])))
            self.fail()
        except AssertionError:
            pass
        c = c.state_replaced(Smoke(Domain([80])))
        numpy.testing.assert_equal(c.smoke.density.data.shape, [1,80,1])

        world = World(add_default_objects=True)
        assert world.gravity.state is world.state.gravity
