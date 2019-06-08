from unittest import TestCase
from phi.data.dataset import *

class TestData(TestCase):

    def test_select(self):
        set = Dataset.load('dir', 'train', range(50))

        set.size(lookup=False)

        set.size(lookup=True)
        self.fail()
