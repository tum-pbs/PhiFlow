from unittest import TestCase

import numpy as np
import torch
import tensorflow

from phi import math
from phi.torch import TORCH_BACKEND


def get_2d_sine(grid_size, L):
    indices = np.array(np.meshgrid(*list(map(range, grid_size))))
    phys_coord = indices.T * L / (grid_size[0])  # between [0, L)
    x, y = phys_coord.T
    d = np.sin(2 * np.pi * x + 1) * np.sin(2 * np.pi * y + 1)
    return d


class TestTorchBackend(TestCase):

    def test_pad(self):
        a = np.random.rand(32, 32)
        a_np = np.pad(a, pad_width=1, mode='wrap')
        a_torch = TORCH_BACKEND.pad(torch.Tensor(a), pad_width=((1, 1), (1, 1)), mode='circular')
        np.array_equal(a_np, a_torch)

    def test_fft_2d(self):
        sine_field = get_2d_sine((32, 32), L=2)
        sine_fft_np = np.fft.fft2(sine_field)
        with math.precision(64):
            batched_sine_field = np.expand_dims(np.expand_dims(sine_field, -1), 0)
            # SciPy backend for reference
            sine_fft_npbe = math.SCIPY_BACKEND.fft(batched_sine_field)
            np.testing.assert_allclose(sine_fft_np, sine_fft_npbe[0, :, :, 0])
            # PyTorch
            sine_fft_torch = TORCH_BACKEND.fft(torch.Tensor(batched_sine_field)).numpy()
            np.testing.assert_allclose(sine_fft_np, sine_fft_torch[0, :, :, 0])

    def test_is_tensor(self):
        self.assertTrue(TORCH_BACKEND.is_tensor(1))
        self.assertTrue(TORCH_BACKEND.is_tensor([0, 1, 2, 3]))
        self.assertTrue(TORCH_BACKEND.is_tensor(torch.zeros(4)))
        self.assertTrue(TORCH_BACKEND.is_tensor(np.zeros(4)))
        self.assertFalse(TORCH_BACKEND.is_tensor(tensorflow.zeros(4)))
        # only native
        self.assertTrue(TORCH_BACKEND.is_tensor(torch.zeros(4), only_native=True))
        self.assertFalse(TORCH_BACKEND.is_tensor(tensorflow.zeros(4), only_native=True))
        self.assertFalse(TORCH_BACKEND.is_tensor([0, 1, 2, 3], only_native=True))
        self.assertFalse(TORCH_BACKEND.is_tensor(np.zeros(4), only_native=True))
