from unittest import TestCase
import numpy as np
from phi.flow import *



class TestWorld(TestCase):

    def test_names(self):
        c = CollectiveState()
        self.assertEqual(c.states, {})
        c = c.state_added(Smoke(Domain([64])))
        try:
            c = c.state_added(Smoke(Domain([80])))
            self.fail()
        except:
            pass
        c = c.state_replaced(Smoke(Domain([80])))
        np.testing.assert_equal(c.smoke.density.data.shape, [1,80,1])

        world = World(add_default_objects=True)
        assert world.gravity.state is world.state.gravity
