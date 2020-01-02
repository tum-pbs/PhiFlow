from numbers import Number

from phi import math
from phi.math.blas import conjugate_gradient
from phi.physics.field import CenteredGrid
from .solver_api import PressureSolver, FluidDomain


# ToDo can cause NaNs, unsafe


class GeometricCG(PressureSolver):

    def __init__(self, accuracy=1e-5, gradient_accuracy='same',
                 max_iterations=2000, max_gradient_iterations='same',
                 autodiff=False):
        '''
        Conjugate gradient solver that geometrically calculates laplace pressure in each iteration.
        Unlike most other solvers, this algorithm is TPU compatible but usually performs worse than SparseCG.
        At the moment, boundary conditions are only partly supported.

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
        '''
        PressureSolver.__init__(self, 'Single-Phase Conjugate Gradient',
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

    def solve(self, divergence, domain, pressure_guess):
        assert isinstance(domain, FluidDomain)
        fluid_mask = domain.accessible_tensor(extend=1)

        if self.autodiff:
            return solve_pressure_forward(divergence, fluid_mask, self.max_iterations, pressure_guess, self.accuracy, domain, back_prop=True)
        else:
            def pressure_gradient(op, grad):
                return solve_pressure_forward(grad, fluid_mask, max_gradient_iterations, None, self.gradient_accuracy, domain)[0]

            pressure, iteration = math.with_custom_gradient(
                solve_pressure_forward,
                [divergence, fluid_mask, self.max_iterations, pressure_guess, self.accuracy, domain],
                pressure_gradient,
                input_index=0, output_index=0, name_base='geom_solve'
            )

            max_gradient_iterations = iteration if self.max_gradient_iterations == 'mirror' else self.max_gradient_iterations
            return pressure, iteration


def solve_pressure_forward(divergence, fluid_mask, max_iterations, guess, accuracy, domain, back_prop=False):

    def apply_A(pressure):
        from phi.physics.material import Material
        mode = 'replicate' if Material.solid(domain.domain.boundaries) else 'constant'
        padded = math.pad(pressure, [[0,0]] + [[1,1]]*(math.ndims(pressure)-2) + [[0,0]], mode=mode)
        return _weighted_sliced_laplace_nd(padded, weights=fluid_mask)

    return conjugate_gradient(divergence, apply_A, guess, accuracy, max_iterations, back_prop=back_prop)


def _weighted_sliced_laplace_nd(tensor, weights):
    if tensor.shape[-1] != 1:
        raise ValueError('Laplace operator requires a scalar channel as input')
    dims = range(math.spatial_rank(tensor))
    components = []
    for dimension in dims:
        center_slices = tuple([(slice(1, -1) if i == dimension else slice(1,-1)) for i in dims])
        upper_slices = tuple([(slice(2, None) if i == dimension else slice(1,-1)) for i in dims])
        lower_slices = tuple([(slice(-2) if i == dimension else slice(1,-1)) for i in dims])

        lower_weights = weights[(slice(None),) + lower_slices + (slice(None),)] * weights[(slice(None),) + center_slices + (slice(None),)]
        upper_weights = weights[(slice(None),) + upper_slices + (slice(None),)] * weights[(slice(None),) + center_slices + (slice(None),)]
        center_weights = - lower_weights - upper_weights

        lower_values = tensor[(slice(None),) + lower_slices + (slice(None),)]
        upper_values = tensor[(slice(None),) + upper_slices + (slice(None),)]
        center_values = tensor[(slice(None),) + center_slices + (slice(None),)]

        diff = math.mul(upper_values, upper_weights) + \
               math.mul(lower_values, lower_weights) + \
               math.mul(center_values, center_weights)
        components.append(diff)
    return math.sum(components, 0)
