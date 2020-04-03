from unittest import TestCase

import numpy as np

from phi.flow import CLOSED, PERIODIC, OPEN, Domain, Noise
from phi.physics.field import CenteredGrid


class TestFieldMath(TestCase):
    
    def test_subtraction_centered_grid(self):
        """subtract one field from another"""
        shape = [32, 27]
        for boundary in [CLOSED, PERIODIC, OPEN]:
            domain = Domain(shape, boundaries=(boundary, boundary))
            centered_grid = CenteredGrid.sample(Noise(), domain)
            result_array = (centered_grid - centered_grid).data
            np.testing.assert_array_equal(result_array, 0)

    def test_addition_centered_grid(self):
        """add one field to another"""
        shape = [32, 27]
        for boundary in [CLOSED, PERIODIC, OPEN]:
            domain = Domain(shape, boundaries=(boundary, boundary))
            centered_grid = CenteredGrid.sample(1, domain)
            result_array = (centered_grid + centered_grid).data
            np.testing.assert_array_equal(result_array, 2)

    def test_multiplication_centered_grid(self):
        """multiply one field with another"""
        shape = [32, 27]
        for boundary in [CLOSED, PERIODIC, OPEN]:
            domain = Domain(shape, boundaries=(boundary, boundary))
            centered_grid = CenteredGrid.sample(1, domain)
            result_array = (centered_grid * centered_grid.copied_with(data=2*np.ones([1] + shape + [1]))).data
            np.testing.assert_array_equal(result_array, 2)

    def test_division_centered_grid(self):
        """divide one field by another"""
        shape = [32, 27]
        for boundary in [CLOSED, PERIODIC, OPEN]:
            domain = Domain(shape, boundaries=(boundary, boundary))
            centered_grid = CenteredGrid.sample(2, domain)
            result_array = (centered_grid / centered_grid.copied_with(data=4*np.ones([1] + shape + [1]))).data
            np.testing.assert_array_equal(result_array, 1./2)
