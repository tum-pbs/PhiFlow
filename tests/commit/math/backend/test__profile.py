from unittest import TestCase

from phi import math
from phi.math.backend import profile


class TestProfile(TestCase):

    def test_profile(self):
        with profile() as prof:
            math.ones() + math.ones()
        prof.print(min_duration=0)
