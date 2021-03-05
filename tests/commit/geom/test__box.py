from unittest import TestCase

from phi import math
from phi.geom import Box


class TestBox(TestCase):

    def test_box_constructor(self):
        box = Box(0, (1, 1))
        math.assert_close(box.size, 1)
        self.assertEqual(math.shape(x=1, y=1), box.shape)

    def test_box_batched(self):
        box = Box(math.tensor([(0, 0), (1, 1)], 'boxes,vector'), 1)
        self.assertEqual(math.shape(boxes=2, x=1, y=1), box.shape)
