from unittest import TestCase
from phi.torch.flow import *
import numpy as np


class TestTorchBackend(TestCase):

    def test_pad(self):
        grid_size = (32, 32)
        a = np.random.rand(*grid_size)
        a_np = np.pad(a, pad_width=1, mode='wrap')
        #a_torch = math.pad(math.tensor(a, names=['x', 'y']), widths={'x': (1, 1), 'y': (1,1)}, mode=PERIODIC)
        a_torch = TORCH_BACKEND.pad(torch.Tensor(a), pad_width=((1, 1), (1, 1)), mode='circular')
        np.array_equal(a_np, a_torch)
