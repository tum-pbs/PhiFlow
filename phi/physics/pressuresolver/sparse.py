import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg

from phi import math
from phi.math.blas import conjugate_gradient
from phi.math.helper import _dim_shifted
from phi.physics.material import Material
from phi.struct.tensorop import collapsed_gather_nd
from .solver_api import PoissonSolver, FluidDomain


class SparseSciPy(PoissonSolver):

    def __init__(self):
        """
        The SciPy solver uses the function scipy.sparse.linalg.spsolve to determine the pressure.
        It does not support initial guesses for the pressure and does not keep track of a loop counter.
        """
        PoissonSolver.__init__(self, 'SciPy sparse solver', supported_devices=('CPU',), supports_guess=False, supports_loop_counter=False, supports_continuous_masks=True)

    def solve(self, field, domain, guess, enable_backprop):
        assert isinstance(domain, FluidDomain)
        dimensions = list(field.shape[1:-1])
        A = sparse_pressure_matrix(dimensions, domain.active_tensor(extend=1), domain.accessible_tensor(extend=1), Material.periodic(domain.domain.boundaries))

        def np_solve_p(div):
            div_vec = div.reshape([-1, A.shape[0]])
            pressure = [scipy.sparse.linalg.spsolve(A, div_vec[i, ...]) for i in range(div_vec.shape[0])]
            return np.array(pressure).reshape(div.shape).astype(np.float32)

        def np_solve_p_gradient(op, grad_in):
            return math.py_func(np_solve_p, [grad_in], np.float32, field.shape)

        pressure = math.py_func(np_solve_p, [field], np.float32, field.shape, grad=np_solve_p_gradient)
        return pressure, None


class SparseCG(PoissonSolver):

    def __init__(self, accuracy=1e-5, max_iterations=2000):
        """
        Conjugate gradient solver using sparse matrix multiplications.

        :param accuracy: the maximally allowed error for each cell, measured in terms of field values.
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
        PoissonSolver.__init__(self, 'Sparse Conjugate Gradient', supported_devices=('CPU', 'GPU'), supports_guess=True, supports_loop_counter=True, supports_continuous_masks=True)
        assert math.is_scalar(accuracy), 'invalid accuracy: %s' % accuracy
        self.accuracy = accuracy
        self.max_iterations = max_iterations

    def solve(self, field, domain, guess, enable_backprop):
        assert isinstance(domain, FluidDomain)
        active_mask = domain.active_tensor(extend=1)
        fluid_mask = domain.accessible_tensor(extend=1)
        dimensions = math.staticshape(field)[1:-1]
        N = int(np.prod(dimensions))
        periodic = Material.periodic(domain.domain.boundaries)

        if math.choose_backend([field, active_mask, fluid_mask]).matches_name('SciPy'):
            A = sparse_pressure_matrix(dimensions, active_mask, fluid_mask, periodic)
        else:
            sidx, sorting = sparse_indices(dimensions, periodic)
            sval_data = sparse_values(dimensions, active_mask, fluid_mask, sorting, periodic)
            backend = math.choose_backend(field)
            sval_data = backend.cast(sval_data, field.dtype)
            A = backend.sparse_tensor(indices=sidx, values=sval_data, shape=[N, N])

        div_vec = math.reshape(field, [-1, int(np.prod(field.shape[1:]))])
        if guess is not None:
            guess = math.reshape(guess, [-1, int(np.prod(field.shape[1:]))])

        def apply_A(pressure): return math.matmul(A, pressure)
        result_vec, iterations = conjugate_gradient(div_vec, apply_A, guess, self.accuracy, self.max_iterations, enable_backprop)
        return math.reshape(result_vec, math.shape(field)), iterations


def sparse_pressure_matrix(dimensions, extended_active_mask, extended_fluid_mask, periodic=False):
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

    diagonal_entries = np.zeros(N, extended_active_mask.dtype)  # diagonal matrix entries

    gridpoints_linear = np.arange(N)
    gridpoints = np.stack(np.unravel_index(gridpoints_linear, dimensions))  # d * (N^2) array mapping from linear to spatial frames

    for dim in dims:
        lower_active, self_active, upper_active = _dim_shifted(extended_active_mask, dim, (-1, 0, 1), diminish_others=(1,1))
        lower_accessible, upper_accessible = _dim_shifted(extended_fluid_mask, dim, (-1, 1), diminish_others=(1, 1))

        stencil_upper = upper_active * self_active
        stencil_lower = lower_active * self_active
        stencil_center = - lower_accessible - upper_accessible

        diagonal_entries += math.flatten(stencil_center)

        dim_direction = math.expand_dims([1 if i == dim else 0 for i in range(d)], axis=-1)
        # --- Stencil upper cells ---
        upper_points, upper_idx = wrap_or_discard(gridpoints + dim_direction, dim, dimensions, periodic=collapsed_gather_nd(periodic, [dim, 1]))
        A[gridpoints_linear[upper_idx], upper_points] = stencil_upper.flatten()[upper_idx]
        # --- Stencil lower cells ---
        lower_points, lower_idx = wrap_or_discard(gridpoints - dim_direction, dim, dimensions, periodic=collapsed_gather_nd(periodic, [dim, 0]))
        A[gridpoints_linear[lower_idx], lower_points] = stencil_lower.flatten()[lower_idx]

    A[gridpoints_linear, gridpoints_linear] = math.minimum(diagonal_entries, -1)  # avoid 0, could lead to NaN

    return scipy.sparse.csc_matrix(A)


def sparse_indices(dimensions, periodic=False):
    N = int(np.prod(dimensions))
    d = len(dimensions)
    dims = range(d)
    gridpoints_linear = np.arange(N)
    gridpoints = np.stack(np.unravel_index(gridpoints_linear, dimensions))  # d * (N^2) array mapping from linear to spatial frames
    indices_list = [np.stack([gridpoints_linear] * 2, axis=-1)]
    for dim in dims:
        dim_direction = math.expand_dims([1 if i == dim else 0 for i in range(d)], axis=-1)
        # --- Stencil upper cells ---
        upper_points, upper_idx = wrap_or_discard(gridpoints + dim_direction, dim, dimensions, periodic=collapsed_gather_nd(periodic, [dim, 1]))
        indices_list.append(np.stack([gridpoints_linear[upper_idx], upper_points], axis=-1))
        # --- Stencil lower cells ---
        lower_points, lower_idx = wrap_or_discard(gridpoints - dim_direction, dim, dimensions, periodic=collapsed_gather_nd(periodic, [dim, 0]))
        indices_list.append(np.stack([gridpoints_linear[lower_idx], lower_points], axis=-1))
    indices = np.concatenate(indices_list, axis=0)
    # --- Sort indices ---
    sorting = np.lexsort(np.transpose(indices)[:, ::-1])
    sorted_indices = indices[sorting]
    return sorted_indices, sorting


def sparse_values(dimensions, extended_active_mask, extended_fluid_mask, sorting=None, periodic=False):
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
    diagonal_entries = 0  # diagonal matrix entries

    gridpoints_linear = np.arange(N)
    gridpoints = np.stack(np.unravel_index(gridpoints_linear, dimensions))  # d * (N^2) array mapping from linear to spatial frames

    for dim in dims:
        lower_active, self_active, upper_active = _dim_shifted(extended_active_mask, dim, (-1, 0, 1), diminish_others=(1, 1))
        lower_accessible, upper_accessible = _dim_shifted(extended_fluid_mask, dim, (-1, 1), diminish_others=(1, 1))

        stencil_upper = upper_active * self_active
        stencil_lower = lower_active * self_active
        stencil_center = - lower_accessible - upper_accessible

        diagonal_entries += math.flatten(stencil_center)

        dim_direction = math.expand_dims([1 if i == dim else 0 for i in range(d)], axis=-1)
        # --- Stencil upper cells ---
        upper_points, upper_idx = wrap_or_discard(gridpoints + dim_direction, dim, dimensions, periodic=collapsed_gather_nd(periodic, [dim, 1]))
        values_list.append(math.gather(math.flatten(stencil_upper), upper_idx))
        # --- Stencil lower cells ---
        lower_points, lower_idx = wrap_or_discard(gridpoints - dim_direction, dim, dimensions, periodic=collapsed_gather_nd(periodic, [dim, 0]))
        values_list.append(math.gather(math.flatten(stencil_lower), lower_idx))

    values_list.insert(0, math.minimum(diagonal_entries, -1.))
    values = math.concat(values_list, axis=0)
    if sorting is not None:
        values = math.gather(values, sorting)
    return values


def wrap_or_discard(points, check_bounds_dim, dimensions, periodic=False):
    """
Handles points that lie outside the domain by either discarding them or wrapping them, depending on periodic.
    :param points: grid indices, typically of shape (dimensions, cell_count)
    :param check_bounds_dim: int
    :param dimensions: domain resolution
    :param periodic: if False: discard indices outside domain, if True: wrap indices outside domain
    :return:
    """
    if not periodic:
        upper_in_range_inx = np.nonzero((points[check_bounds_dim] < dimensions[check_bounds_dim]) & (points[check_bounds_dim] >= 0))[0]
        new_points = points[:, upper_in_range_inx]  # discard points outside domain
    else:
        upper_in_range_inx = slice(None)
        new_points = points % math.expand_dims(dimensions, -1)  # wrap points

    linear_points = np.ravel_multi_index(new_points, dimensions)
    return linear_points, upper_in_range_inx
