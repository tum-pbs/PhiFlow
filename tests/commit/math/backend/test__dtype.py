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
        self.assertEqual(from_numpy_dtype(np.array(0, np.object).dtype), DType(object))

    def test_object_dtype(self):
        self.assertIn(DType(object).bits, (32, 64))

    def test_as_dtype(self):
        self.assertEqual(None, DType.as_dtype(None))
        self.assertEqual(DType(int, 32), DType.as_dtype(DType(int, 32)))
        self.assertEqual(DType(int, 32), DType.as_dtype(int))
        self.assertEqual(DType(float, 32), DType.as_dtype(float))
        self.assertEqual(DType(complex, 64), DType.as_dtype(complex))
        self.assertEqual(DType(bool), DType.as_dtype(bool))
        self.assertEqual(DType(int, 8), DType.as_dtype((int, 8)))
        self.assertEqual(object, DType.as_dtype(object).kind)
        try:
            DType.as_dtype(str)
            self.fail()
        except ValueError:
            pass

