import logging
from numbers import Number
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg

from phi import math
from phi.math.blas import conjugate_gradient
from .solver_api import PressureSolver, FluidDomain


class SparseSciPy(PressureSolver):

    def __init__(self):
        """
        The SciPy solver uses the function scipy.sparse.linalg.spsolve to determine the pressure.
        It does not support initial guesses for the pressure and does not keep track of a loop counter.
        """
        PressureSolver.__init__(self, 'SciPy sparse solver',
                                supported_devices=('CPU',),
                                supports_guess=False, supports_loop_counter=False, supports_continuous_masks=True)

    def solve(self, divergence, domain, pressure_guess):
        assert isinstance(domain, FluidDomain)
        dimensions = list(divergence.shape[1:-1])
        A = sparse_pressure_matrix(dimensions, domain.active_tensor(extend=1), domain.accessible_tensor(extend=1))

        def np_solve_p(div):
            div_vec = div.reshape([-1, A.shape[0]])
            pressure = [scipy.sparse.linalg.spsolve(A, div_vec[i, ...]) for i in range(div_vec.shape[0])]
            return np.array(pressure).reshape(div.shape).astype(np.float32)

        def np_solve_p_gradient(op, grad_in):
            return math.py_func(np_solve_p, [grad_in], np.float32, divergence.shape)

        pressure = math.py_func(np_solve_p, [divergence], np.float32, divergence.shape, grad=np_solve_p_gradient)
        return pressure, None


def sparse_pressure_matrix(dimensions, extended_active_mask, extended_fluid_mask):
    """
    Builds a sparse matrix such that when applied to a flattened pressure channel, it calculates the laplace
    of that channel, taking into account obstacles and empty cells.

    :param dimensions: valid simulation dimensions. Pressure channel should be of shape (batch size, dimensions..., 1)
    :param extended_active_mask: Binary tensor with 2 more entries in every dimension than 'dimensions'.
    :param extended_fluid_mask: Binary tensor with 2 more entries in every dimension than 'dimensions'.
    :return: SciPy sparse matrix that acts as a laplace on a flattened pressure channel given obstacles and empty cells
    """
    N = int(np.prod(dimensions))
    d = len(dimensions)
    A = scipy.sparse.lil_matrix((N, N), dtype=np.float32)
    dims = range(d)

    center_values = None # diagonal matrix entries

    gridpoints_linear = np.arange(N)
    gridpoints = np.stack(np.unravel_index(gridpoints_linear, dimensions))  # d * (N^2) array mapping from linear to spatial frames

    for dim in dims:
        upper_indices = tuple([slice(None)] + [slice(2, None) if i == dim else slice(1, -1) for i in dims] + [slice(None)])
        center_indices = tuple([slice(None)] + [slice(1, -1) if i == dim else slice(1, -1) for i in dims] + [slice(None)])
        lower_indices = tuple([slice(None)] + [slice(0, -2) if i == dim else slice(1, -1) for i in dims] + [slice(None)])

        self_active = extended_active_mask[center_indices]
        stencil_upper = extended_active_mask[upper_indices] * self_active
        stencil_lower = extended_active_mask[lower_indices] * self_active
        stencil_center = - extended_fluid_mask[upper_indices] - extended_fluid_mask[lower_indices]

        if center_values is None:
            center_values = math.flatten(stencil_center)
        else:
            center_values = center_values + math.flatten(stencil_center)

        # Find entries in matrix
        dim_direction = np.zeros_like(gridpoints)
        dim_direction[dim] = 1
        # Upper frames
        upper_indices = gridpoints + dim_direction
        upper_in_range_inx = np.nonzero(upper_indices[dim] < dimensions[dim])
        upper_indices_linear = np.ravel_multi_index(upper_indices[:, upper_in_range_inx], dimensions)
        A[gridpoints_linear[upper_in_range_inx], upper_indices_linear] = stencil_upper.flatten()[upper_in_range_inx]
        # Lower frames
        lower_indices = gridpoints - dim_direction
        lower_in_range_inx = np.nonzero(lower_indices[dim] >= 0)
        lower_indices_linear = np.ravel_multi_index(lower_indices[:, lower_in_range_inx], dimensions)
        A[gridpoints_linear[lower_in_range_inx], lower_indices_linear] = stencil_lower.flatten()[lower_in_range_inx]

    A[gridpoints_linear, gridpoints_linear] = math.minimum(center_values, -1)

    return scipy.sparse.csc_matrix(A)


class SparseCG(PressureSolver):

    def __init__(self, accuracy=1e-5, gradient_accuracy='same',
                 max_iterations=2000, max_gradient_iterations='same',
                 autodiff=False):
        """
        Conjugate gradient solver using sparse matrix multiplications.

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
        PressureSolver.__init__(self, 'Sparse Conjugate Gradient',
                                supported_devices=('CPU', 'GPU'),
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
        active_mask = domain.active_tensor(extend=1)
        fluid_mask = domain.accessible_tensor(extend=1)
        dimensions = list(divergence.shape[1:-1])
        N = int(np.prod(dimensions))

        if math.choose_backend(divergence).matches_name('SciPy'):
            A = sparse_pressure_matrix(dimensions, active_mask, fluid_mask)
        else:
            sidx, sorting = sparse_indices(dimensions)
            sval_data = sparse_values(dimensions, active_mask, fluid_mask, sorting)
            A = math.choose_backend(divergence).sparse_tensor(indices=sidx, values=sval_data, shape=[N, N])

        if self.autodiff:
            return sparse_cg(divergence, A, self.max_iterations, pressure_guess, self.accuracy, back_prop=True)
        else:
            def pressure_gradient(op, grad):
                return sparse_cg(grad, A, max_gradient_iterations, None, self.gradient_accuracy)[0]

            pressure, iteration = math.with_custom_gradient(sparse_cg,
                                                            [divergence, A, self.max_iterations, pressure_guess, self.accuracy],
                                                            pressure_gradient, input_index=0, output_index=0,
                                                            name_base='scg_pressure_solve')

            max_gradient_iterations = iteration if self.max_gradient_iterations == 'mirror' else self.max_gradient_iterations
            return pressure, iteration


def sparse_cg(divergence, A, max_iterations, guess, accuracy, back_prop=False):
    div_vec = math.reshape(divergence, [-1, int(np.prod(divergence.shape[1:]))])
    if guess is not None:
        guess = math.reshape(guess, [-1, int(np.prod(divergence.shape[1:]))])
    apply_A = lambda pressure: math.matmul(A, pressure)
    result_vec, iterations = conjugate_gradient(div_vec, apply_A, guess, accuracy, max_iterations, back_prop)
    return math.reshape(result_vec, math.shape(divergence)), iterations


def sparse_indices(dimensions):
    N = int(np.prod(dimensions))
    d = len(dimensions)
    dims = range(d)

    gridpoints_linear = np.arange(N)
    gridpoints = np.stack(np.unravel_index(gridpoints_linear, dimensions)) # d * (N^2) array mapping from linear to spatial frames

    indices_list = [np.stack([gridpoints_linear] * 2, axis=-1)]

    for dim in dims:
        dim_direction = np.zeros_like(gridpoints)
        dim_direction[dim] = 1
        # Upper frames
        upper_indices = gridpoints + dim_direction
        upper_in_range_inx = np.nonzero(upper_indices[dim] < dimensions[dim])
        upper_indices_linear = np.ravel_multi_index(upper_indices[:, upper_in_range_inx], dimensions)[0, :]
        indices_list.append(np.stack([gridpoints_linear[upper_in_range_inx], upper_indices_linear], axis=-1))
        # Lower frames
        lower_indices = gridpoints - dim_direction
        lower_in_range_inx = np.nonzero(lower_indices[dim] >= 0)
        lower_indices_linear = np.ravel_multi_index(lower_indices[:, lower_in_range_inx], dimensions)[0, :]
        indices_list.append(np.stack([gridpoints_linear[lower_in_range_inx], lower_indices_linear], axis=-1))

    indices = np.concatenate(indices_list, axis=0)

    sorting = np.lexsort(np.transpose(indices)[:, ::-1])

    sorted_indices = indices[sorting]

    return sorted_indices, sorting


def sparse_values(dimensions, extended_active_mask, extended_fluid_mask, sorting=None):
    """
    Builds a sparse matrix such that when applied to a flattened pressure channel, it calculates the laplace
    of that channel, taking into account obstacles and empty cells.

    :param dimensions: valid simulation dimensions. Pressure channel should be of shape (batch size, dimensions..., 1)
    :param extended_active_mask: Binary tensor with 2 more entries in every dimension than 'dimensions'.
    :param extended_fluid_mask: Binary tensor with 2 more entries in every dimension than 'dimensions'.
    :return: SciPy sparse matrix that acts as a laplace on a flattened pressure channel given obstacles and empty cells
    """
    N = int(np.prod(dimensions))
    d = len(dimensions)
    dims = range(d)

    values_list = []
    center_values = None # diagonal matrix entries

    gridpoints_linear = np.arange(N)
    gridpoints = np.stack(np.unravel_index(gridpoints_linear, dimensions)) # d * (N^2) array mapping from linear to spatial frames

    for dim in dims:
        upper_indices = tuple([slice(None)] + [slice(2, None) if i == dim else slice(1, -1) for i in dims] + [slice(None)])
        center_indices = tuple([slice(None)] + [slice(1, -1) if i == dim else slice(1, -1) for i in dims] + [slice(None)])
        lower_indices = tuple([slice(None)] + [slice(0, -2) if i == dim else slice(1, -1) for i in dims] + [slice(None)])

        self_active = extended_active_mask[center_indices]
        stencil_upper = extended_active_mask[upper_indices] * self_active
        stencil_lower = extended_active_mask[lower_indices] * self_active
        stencil_center = - extended_fluid_mask[upper_indices] - extended_fluid_mask[lower_indices]

        if center_values is None:
            center_values = math.flatten(stencil_center)
        else:
            center_values = center_values + math.flatten(stencil_center)

        dim_direction = np.zeros_like(gridpoints)
        dim_direction[dim] = 1
        # Upper frames
        upper_indices = gridpoints + dim_direction
        upper_in_range_inx = np.nonzero(upper_indices[dim] < dimensions[dim])[0]
        values_list.append(math.gather(math.flatten(stencil_upper), upper_in_range_inx))
        # Lower frames
        lower_indices = gridpoints - dim_direction
        lower_in_range_inx = np.nonzero(lower_indices[dim] >= 0)[0]
        values_list.append(math.gather(math.flatten(stencil_lower), lower_in_range_inx))

    center_values = math.minimum(center_values, -1.)
    values_list.insert(0, center_values)

    values = math.concat(values_list, axis=0)
    if sorting is not None:
        values = math.gather(values, sorting)
    return values
