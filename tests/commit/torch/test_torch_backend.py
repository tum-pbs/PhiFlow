from unittest import TestCase

import numpy as np
import torch

from phi.torch import TORCH_BACKEND


def get_2d_sine(grid_size, L):
    indices = np.array(np.meshgrid(*list(map(range, grid_size))))
    phys_coord = indices.T * L / (grid_size[0])  # between [0, L)
    x, y = phys_coord.T
    d = np.sin(2 * np.pi * x + 1) * np.sin(2 * np.pi * y + 1)
    return d


class TestTorchBackend(TestCase):

    def test_pad(self):
        grid_size = (32, 32)
        a = np.random.rand(*grid_size)
        a_np = np.pad(a, pad_width=1, mode='wrap')
        #a_torch = math.pad(math.tensor(a, names=['x', 'y']), widths={'x': (1, 1), 'y': (1,1)}, mode=PERIODIC)
        a_torch = TORCH_BACKEND.pad(torch.Tensor(a), pad_width=((1, 1), (1, 1)), mode='circular')
        np.array_equal(a_np, a_torch)

    def test_fft_2d(self):
        grid_size = (32, 32)
        N = grid_size[0]
        L = 2
        sine_field = get_2d_sine((N, N), L=L)
        sine_fft_np = np.fft.fft2(sine_field)
        sine_fft_torch = TORCH_BACKEND.fft(torch.Tensor(sine_field))
        np.array_equal(sine_fft_np, sine_fft_torch.numpy())
