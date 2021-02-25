from unittest import TestCase

import numpy as np
import tensorflow as tf
import torch

from phi.tf import TF_BACKEND


class TestNothing(TestCase):

    def test_is_tensor(self):
        self.assertTrue(TF_BACKEND.is_tensor(1))
        self.assertTrue(TF_BACKEND.is_tensor([0, 1, 2, 3]))
        self.assertTrue(TF_BACKEND.is_tensor(tf.zeros(4)))
        self.assertTrue(TF_BACKEND.is_tensor(np.zeros(4)))
        self.assertFalse(TF_BACKEND.is_tensor(torch.zeros(4)))
        # only native
        self.assertTrue(TF_BACKEND.is_tensor(tf.zeros(4), only_native=True))
        self.assertFalse(TF_BACKEND.is_tensor(torch.zeros(4), only_native=True))
        self.assertFalse(TF_BACKEND.is_tensor([0, 1, 2, 3], only_native=True))
        self.assertFalse(TF_BACKEND.is_tensor(np.zeros(4), only_native=True))
