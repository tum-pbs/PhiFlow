from unittest import TestCase

import numpy as np
import tensorflow as tf
import torch

from phi.jax import JAX
from jax import numpy as jnp


class TestNothing(TestCase):

    def test_is_tensor(self):
        self.assertTrue(JAX.is_tensor(1))
        self.assertTrue(JAX.is_tensor([0, 1, 2, 3]))
        self.assertTrue(JAX.is_tensor(jnp.zeros(4)))
        self.assertTrue(JAX.is_tensor(np.zeros(4)))
        self.assertFalse(JAX.is_tensor(torch.zeros(4)))
        # only native
        self.assertTrue(JAX.is_tensor(jnp.zeros(4), only_native=True))
        self.assertFalse(JAX.is_tensor(torch.zeros(4), only_native=True))
        self.assertFalse(JAX.is_tensor([0, 1, 2, 3], only_native=True))
        self.assertFalse(JAX.is_tensor(np.zeros(4), only_native=True))
        # Others
        self.assertFalse(JAX.is_tensor('string'))
