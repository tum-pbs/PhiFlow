from unittest import TestCase

import numpy
import six

from phi import struct
from phi.physics.collective import StateCollection
from phi.physics.domain import Domain
from phi.physics.fluid import Fluid
from phi.physics.world import World


class TestWorld(TestCase):

    def test_names(self):
        c = StateCollection()
        self.assertEqual(c.states, {})
        c = c.state_added(Fluid(Domain([64])))
        try:
            c = c.state_added(Fluid(Domain([80])))
            self.fail()
        except AssertionError:
            pass
        c = c.state_replaced(Fluid(Domain([80])))
        numpy.testing.assert_equal(c.fluid.density.data.shape, [1,80,1])

        world = World(add_default_objects=True)
        assert world.gravity.state is world.state.gravity

    def test_state_collection(self):
        fluid = Fluid(Domain([1, 1]))
        fluid2 = Fluid(Domain([2, 2]))

        c1 = StateCollection([fluid])
        assert c1.fluid is fluid
        assert fluid in c1
        assert c1[fluid] is fluid
        assert isinstance(repr(c1), six.string_types)
        assert len(c1) == len(c1.shape) == len(c1.staticshape) == len(c1.dtype)
        assert c1.shape.fluid.density.data == (1, 1, 1, 1)
        self.assertIsInstance(c1.dtype.fluid.density.data, numpy.dtype)

        c2 = StateCollection()
        assert len(c2) == 0
        c2 = c2.state_added(fluid)
        assert c2 == c1
        assert hash(c2) == hash(c1)

        c3 = c2.state_replaced(fluid2)
        assert c3 != c2
        assert c3.fluid is fluid2

        c4 = c3.state_removed(fluid2)
        assert len(c4) == 0

        c5 = struct.map(lambda x: x, c1)
        assert isinstance(c5, StateCollection)
        assert c5 == c1
