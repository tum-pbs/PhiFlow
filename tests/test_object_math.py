from unittest import TestCase

import numpy as np
from phi import math

from phi.flow import CLOSED, PERIODIC, OPEN, Domain, poisson_solve, Noise
from phi.physics.pressuresolver.geom import GeometricCG
from phi.physics.pressuresolver.sparse import SparseCG, SparseSciPy
from phi.physics.pressuresolver.fourier import Fourier
from phi.physics.field import CenteredGrid
from phi.geom.geometry import AABox
from phi.physics.field import Field


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
        """add one field from another"""
        shape = [32, 27]
        for boundary in [CLOSED, PERIODIC, OPEN]:
            domain = Domain(shape, boundaries=(boundary, boundary))
            centered_grid = CenteredGrid.sample(Noise(), domain)
            centered_grid = centered_grid.copied_with(data=np.ones([1] + shape + [1]))
            #centered_grid = Field(np.ones(shape)).sample_at(shape)#centered_grid.copied_with(data=np.ones(shape))
            result_array = (centered_grid + centered_grid).data
            np.testing.assert_array_equal(result_array, 2)

    def test_multiplication_centered_grid(self):
        """multiply one field from another"""
        shape = [32, 27]
        for boundary in [CLOSED, PERIODIC, OPEN]:
            domain = Domain(shape, boundaries=(boundary, boundary))
            centered_grid = CenteredGrid.sample(Noise(), domain)
            centered_grid = centered_grid.copied_with(data=np.ones([1] + shape + [1]))
            result_array = (centered_grid * centered_grid.copied_with(data=2*np.ones([1] + shape + [1]))).data
            np.testing.assert_array_equal(result_array, 2)

    def test_division_centered_grid(self):
        """divide one field from another"""
        shape = [32, 27]
        for boundary in [CLOSED, PERIODIC, OPEN]:
            domain = Domain(shape, boundaries=(boundary, boundary))
            centered_grid = CenteredGrid.sample(Noise(), domain)
            centered_grid = centered_grid.copied_with(data=2*np.ones([1] + shape + [1]))
            result_array = (centered_grid / centered_grid.copied_with(data=4*np.ones([1] + shape + [1]))).data
            np.testing.assert_array_equal(result_array, 1./2)
