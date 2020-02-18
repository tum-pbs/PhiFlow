from numbers import Number

from phi import math
from phi.math.blas import conjugate_gradient
from phi.math.helper import _dim_shifted
from phi.physics.field import CenteredGrid
from .solver_api import PoissonDomain, PoissonSolver


class GeometricCG(PoissonSolver):

    def __init__(self, accuracy=1e-5, gradient_accuracy='same',
                 max_iterations=2000, max_gradient_iterations='same',
                 autodiff=False):
        """
Conjugate gradient solver that geometrically calculates laplace pressure in each iteration.
Unlike most other solvers, this algorithm is TPU compatible but usually performs worse than SparseCG.

Obstacles are allowed to vary between examples but the same number of iterations is performed for each example in one batch.

        :param accuracy: the maximally allowed error on the divergence channel for each cell
        :param gradient_accuracy: accuracy applied during backpropagation, number of 'same' to use forward accuracy
        :param max_iterations: integer specifying maximum conjugent gradient loop iterations or None for no limit
        :param max_gradient_iterations: maximum loop iterations during backpropagation,
            'same' uses the number from max_iterations,
            'mirror' sets the maximum to the number of iterations that were actually performed in the forward pass
        :param autodiff: If autodiff=True, use the built-in autodiff for backpropagation.
            The intermediate results of each loop iteration will be permanently stored if backpropagation is used.
            If False, replaces autodiff by a forward pressure solve in reverse accumulation backpropagation.
            This requires less memory but is only accurate if the solution is fully converged.
        """
        PoissonSolver.__init__(self, 'Single-Phase Conjugate Gradient',
                               supported_devices=('CPU', 'GPU', 'TPU'),
                               supports_guess=True, supports_loop_counter=True, supports_continuous_masks=True)
        assert isinstance(accuracy, Number), 'invalid accuracy: %s' % accuracy
        assert gradient_accuracy == 'same' or isinstance(gradient_accuracy, Number), 'invalid gradient_accuracy: %s' % gradient_accuracy
        assert max_gradient_iterations in ['same', 'mirror'] or isinstance(max_gradient_iterations, Number), 'invalid max_gradient_iterations: %s' % max_gradient_iterations
        self.accuracy = accuracy
        self.gradient_accuracy = accuracy if gradient_accuracy == 'same' else gradient_accuracy
        self.max_iterations = max_iterations
        if max_gradient_iterations == 'same':
            self.max_gradient_iterations = max_iterations
        elif max_gradient_iterations == 'mirror':
            self.max_gradient_iterations = 'mirror'
        else:
            self.max_gradient_iterations = max_gradient_iterations
            assert not autodiff, 'Cannot specify max_gradient_iterations when autodiff=True'
        self.autodiff = autodiff

    def solve(self, divergence, domain, guess):
        assert isinstance(domain, PoissonDomain)
        fluid_mask = domain.accessible_tensor(extend=1)

        if self.autodiff:
            return solve_pressure_forward(divergence, fluid_mask, self.max_iterations, guess, self.accuracy, domain, back_prop=True)
        else:
            def pressure_gradient(op, grad):
                return solve_pressure_forward(grad, fluid_mask, max_gradient_iterations, None, self.gradient_accuracy, domain)[0]

            pressure, iteration = math.with_custom_gradient(
                solve_pressure_forward,
                [divergence, fluid_mask, self.max_iterations, guess, self.accuracy, domain],
                pressure_gradient,
                input_index=0, output_index=0, name_base='geom_solve'
            )

            max_gradient_iterations = iteration if self.max_gradient_iterations == 'mirror' else self.max_gradient_iterations
            return pressure, iteration


def solve_pressure_forward(divergence, fluid_mask, max_iterations, guess, accuracy, domain, back_prop=False):
    from phi.physics.material import Material
    extrapolation = Material.extrapolation_mode(domain.domain.boundaries)

    def apply_A(pressure):
        pressure = CenteredGrid(pressure, extrapolation=extrapolation)
        pressure_padded = pressure.padded([[1, 1]] * pressure.rank)
        return _weighted_sliced_laplace_nd(pressure_padded.data, weights=fluid_mask)

    return conjugate_gradient(divergence, apply_A, guess, accuracy, max_iterations, back_prop=back_prop)


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
