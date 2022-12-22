from unittest import TestCase

import numpy as np

from phi.math import DType
from phi.math.backend._dtype import from_numpy_dtype, combine_types


class TestDType(TestCase):

    def test_from_numpy_dtype(self):
        self.assertEqual(from_numpy_dtype(bool), DType(bool))
        self.assertEqual(from_numpy_dtype(np.bool_), DType(bool))
        self.assertEqual(from_numpy_dtype(np.int32), DType(int, 32))
        self.assertEqual(from_numpy_dtype(np.array(0, np.int32).dtype), DType(int, 32))
        self.assertEqual(from_numpy_dtype(np.array(0, bool).dtype), DType(bool))
        self.assertEqual(from_numpy_dtype(np.array(0, object).dtype), DType(object))

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

    def test_create_by_precision(self):
        self.assertEqual(DType(float, precision=16), DType(float, 16))
        self.assertEqual(DType(complex, precision=32), DType(complex, 64))
        try:
            DType(bool, precision=16)
            raise RuntimeError
        except AssertionError:
            pass
        try:
            DType(int, precision=16)
            raise RuntimeError
        except AssertionError:
            pass
        try:
            DType(object, precision=16)
            raise RuntimeError
        except AssertionError:
            pass

    def test_combine_types(self):
        self.assertEqual(DType(float, 64), combine_types(DType(float, 32), DType(float, 64)))
        self.assertEqual(DType(float, 32), combine_types(DType(float, 32), DType(int, 64)))
        self.assertEqual(DType(int, 32), combine_types(DType(int, 32), DType(int, 16)))
        self.assertEqual(DType(complex, 128), combine_types(DType(complex, 32), DType(float, 64)))
