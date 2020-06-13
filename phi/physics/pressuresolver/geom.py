from numbers import Number

from phi import math
from phi.math.blas import conjugate_gradient
from phi.math.helper import _dim_shifted
from phi.physics.field import CenteredGrid
from .solver_api import PoissonDomain, PoissonSolver
from phi.physics.material import Material


class GeometricCG(PoissonSolver):

    def __init__(self, accuracy=1e-5, max_iterations=2000):
        """
Conjugate gradient solver that geometrically calculates laplace pressure in each iteration.
Unlike most other solvers, this algorithm is TPU compatible but usually performs worse than SparseCG.

Obstacles are allowed to vary between examples but the same number of iterations is performed for each example in one batch.

        :param accuracy: the maximally allowed error on the divergence channel for each cell
        :param max_iterations: integer specifying maximum conjugent gradient loop iterations or None for no limit
        :param autodiff:
        """
        PoissonSolver.__init__(self, 'Single-Phase Conjugate Gradient', supported_devices=('CPU', 'GPU', 'TPU'), supports_guess=True, supports_loop_counter=True, supports_continuous_masks=True)
        assert math.is_scalar(accuracy), 'invalid accuracy: %s' % accuracy
        self.accuracy = accuracy
        self.max_iterations = max_iterations

    def solve(self, divergence, domain, guess, enable_backprop):
        assert isinstance(domain, PoissonDomain)
        fluid_mask = domain.accessible_tensor(extend=1)
        extrapolation = Material.extrapolation_mode(domain.domain.boundaries)

        def apply_A(pressure):
            pressure = CenteredGrid(pressure, extrapolation=extrapolation)
            pressure_padded = pressure.padded([[1, 1]] * pressure.rank)
            return _weighted_sliced_laplace_nd(pressure_padded.data, weights=fluid_mask)

        return conjugate_gradient(divergence, apply_A, guess, self.accuracy, self.max_iterations, back_prop=enable_backprop)


def _weighted_sliced_laplace_nd(tensor, weights):
    if tensor.shape[-1] != 1:
        raise ValueError('Laplace operator requires a scalar channel as input')
    dims = range(math.spatial_rank(tensor))
    components = []
    for dimension in dims:
        lower_weights, center_weights, upper_weights = _dim_shifted(weights, dimension, (-1, 0, 1), diminish_others=(1, 1))
        lower_values, center_values, upper_values = _dim_shifted(tensor, dimension, (-1, 0, 1), diminish_others=(1, 1))
        diff = math.mul(upper_values, upper_weights * center_weights) + math.mul(lower_values, lower_weights * center_weights) + math.mul(center_values, - lower_weights - upper_weights)
        components.append(diff)
    return math.sum(components, 0)
