from unittest import TestCase

import numpy as np
import tensorflow as tf
import torch

from phi.tf import TENSORFLOW


class TestNothing(TestCase):

    def test_is_tensor(self):
        self.assertTrue(TENSORFLOW.is_tensor(1))
        self.assertTrue(TENSORFLOW.is_tensor([0, 1, 2, 3]))
        self.assertTrue(TENSORFLOW.is_tensor(tf.zeros(4)))
        self.assertTrue(TENSORFLOW.is_tensor(np.zeros(4)))
        self.assertFalse(TENSORFLOW.is_tensor(torch.zeros(4)))
        # only native
        self.assertTrue(TENSORFLOW.is_tensor(tf.zeros(4), only_native=True))
        self.assertFalse(TENSORFLOW.is_tensor(torch.zeros(4), only_native=True))
        self.assertFalse(TENSORFLOW.is_tensor([0, 1, 2, 3], only_native=True))
        self.assertFalse(TENSORFLOW.is_tensor(np.zeros(4), only_native=True))
        # Others
        self.assertFalse(TENSORFLOW.is_tensor('string'))
