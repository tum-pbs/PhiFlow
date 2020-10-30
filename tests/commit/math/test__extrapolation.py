from unittest import TestCase

import phi.math as math
from phi.math._extrapolation import Extrapolation


class TestPass(TestCase):

    def test_pass(self):
        pass


class TestExtrapolation(TestCase):

    def test_pad(self):
        test_in_func_out = [
            (math.zeros(x=3, y=4, z=5, a=1),
             lambda tensor: ConstantExtrapolation(0).pad(tensor, dict(x=[1, 1], y=[1, 0], z=[0, 1], a=[0, 0]))
             math.zeros(x=5, y=5, z=6, a=1)),
            (math.ones(x=3, y=4, z=5, a=1),
             lambda tensor: ConstantExtrapolation(1).pad(tensor, dict(x=[1, 1], y=[1, 0], z=[0, 1], a=[0, 0]))
             math.ones(x=5, y=5, z=6, a=1)),
            (-math.ones(x=3, y=4, z=5, a=1),
             lambda tensor: ConstantExtrapolation(-1).pad(tensor, dict(x=[1, 1], y=[1, 0], z=[0, 1], a=[0, 0]))
             - math.ones(x=5, y=5, z=6, a=1)),
        ]
        for val_in, func, val_out in test_in_func_out:
            try:
                math.assert_equal(val_out, func(val_in))
            except Exception as e:
                raise BaseException(AssertionError(e, val_in, func, val_out))
