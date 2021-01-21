from unittest import TestCase

import numpy as np

from phi.math import DType
from phi.math.backend._dtype import from_numpy_dtype


class TestDType(TestCase):

    def test_from_numpy_dtype(self):
        self.assertEqual(from_numpy_dtype(np.bool), DType(bool))
        self.assertEqual(from_numpy_dtype(np.bool_), DType(bool))
        self.assertEqual(from_numpy_dtype(np.int32), DType(int, 32))
        self.assertEqual(from_numpy_dtype(np.array(0, np.int32).dtype), DType(int, 32))
        self.assertEqual(from_numpy_dtype(np.array(0, bool).dtype), DType(bool))
