from functools import partial
from typing import Tuple, Callable

import numpy as np

from ._backend import Backend, SolveResult, List, DType, spatial_derivative_evaluation


def identity(x):
    return x


def stop_on_l2(b: Backend, tolerance_squared, max_iter: np.ndarray, on_diverged: Exception or None = None):
    max_iter = b.as_tensor(max_iter[-1, :])
    rsq0 = []
    def check_progress(iterations, residual_squared):
        converged = b.all(residual_squared <= tolerance_squared, axis=(1,))
        if not rsq0:
            diverged = b.any(~b.isfinite(residual_squared), axis=(1,))
            rsq0.append(residual_squared)
        else:
            diverged = b.any(residual_squared / rsq0[0] > 1e5, axis=(1,)) & (iterations >= 8)
        continue_ = ~converged & ~diverged & (iterations < max_iter)
        if on_diverged is not None and b.any(diverged):
            on_diverged(iterations)
        return continue_, converged, diverged
    return check_progress


def _max_iter(max_iter: np.ndarray) -> int or list:
    trj_size, batch_size = max_iter.shape
    if trj_size == 1:
        return int(np.max(max_iter))
    else:
        assert np.all(max_iter == max_iter[:, :1]), "When recording a trajectory, max_iter must be equal for all batch entries"
        return max_iter[:, 0].tolist()


def cg(b: Backend, lin, y, x0, check_progress: Callable, max_iter, pre: Callable = identity) -> SolveResult or List[SolveResult]:
    """
    Based on "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain" by Jonathan Richard Shewchuk
    symbols: dx=d, dy=q, step_size=alpha, residual_squared=delta, residual=r, y=b, pre=M
    """
    batch_size = b.staticshape(y)[0]
    y = b.to_float(y)
    x = b.copy(b.to_float(x0), only_mutable=True)
    residual = y - b.linear(lin, x)
    dx = pre(residual)
    iterations = b.zeros([batch_size], DType(int, 32))
    function_evaluations = b.ones([batch_size], DType(int, 32))
    residual_squared = b.sum(residual * dx, -1, keepdims=True)
    continue_, converged, diverged = check_progress(iterations, residual_squared)

    def cg_loop_body(continue_, x, dx, residual_squared, residual, iterations, function_evaluations, _converged, _diverged):
        continue_1 = b.to_int32(continue_)
        iterations += continue_1
        with spatial_derivative_evaluation(1):
            dy = b.linear(lin, dx); function_evaluations += continue_1
        dx_dy = b.sum(dx * dy, axis=-1, keepdims=True)
        step_size = b.divide_no_nan(residual_squared, dx_dy)
        step_size *= b.expand_dims(b.to_float(continue_1), -1)  # this is not really necessary but ensures batch-independence
        x += step_size * dx
        residual = residual - step_size * dy  # in-place subtraction affects convergence
        s = pre(residual)
        residual_squared_old = residual_squared
        residual_squared = b.sum(residual * s, -1, keepdims=True)
        dx = s + b.divide_no_nan(residual_squared, residual_squared_old) * dx
        continue_, converged, diverged = check_progress(iterations, residual_squared)
        return continue_, x, dx, residual_squared, residual, iterations, function_evaluations, converged, diverged

    _, x, _, _, residual, iterations, function_evaluations, converged, diverged = b.while_loop(cg_loop_body, (continue_, x, dx, residual_squared, residual, iterations, function_evaluations, converged, diverged), _max_iter(max_iter))
    return SolveResult(f"Φ-Flow CG ({b.name})", x, residual, iterations, function_evaluations, converged, diverged, [""] * batch_size)


def cg_adaptive(b, lin, y, x0, check_progress: Callable, max_iter) -> SolveResult or List[SolveResult]:
    """
    Based on the variant described in "Methods of Conjugate Gradients for Solving Linear Systems" by Magnus R. Hestenes and Eduard Stiefel https://nvlpubs.nist.gov/nistpubs/jres/049/jresv49n6p409_A1b.pdf
    """
    y = b.to_float(y)
    x0 = b.copy(b.to_float(x0), only_mutable=True)
    batch_size = b.staticshape(y)[0]
    x = x0
    dx = residual = y - b.linear(lin, x)
    dy = b.linear(lin, dx)
    iterations = b.zeros([batch_size], DType(int, 32))
    function_evaluations = b.ones([batch_size], DType(int, 32))
    residual_squared = b.sum(residual ** 2, -1, keepdims=True)
    continue_, converged, diverged = check_progress(iterations, residual_squared)

    def acg_loop_body(continue_, x, dx, dy, residual, iterations, function_evaluations, _converged, _diverged):
        continue_1 = b.to_int32(continue_)
        iterations += continue_1
        dx_dy = b.sum(dx * dy, axis=-1, keepdims=True)
        step_size = b.divide_no_nan(b.sum(dx * residual, axis=-1, keepdims=True), dx_dy)
        step_size *= b.expand_dims(b.to_float(continue_1), -1)  # this is not really necessary but ensures batch-independence
        x += step_size * dx
        residual = residual - step_size * dy  # in-place subtraction affects convergence
        residual_squared = b.sum(residual ** 2, -1, keepdims=True)
        dx = residual - b.divide_no_nan(b.sum(residual * dy, axis=-1, keepdims=True) * dx, dx_dy)
        with spatial_derivative_evaluation(1):
            dy = b.linear(lin, dx); function_evaluations += continue_1
        continue_, converged, diverged = check_progress(iterations, residual_squared)
        return continue_, x, dx, dy, residual, iterations, function_evaluations, converged, diverged

    _, x, _, _, residual, iterations, function_evaluations, converged, diverged = b.while_loop(acg_loop_body, (continue_, x, dx, dy, residual, iterations, function_evaluations, converged, diverged), _max_iter(max_iter))
    return SolveResult(f"Φ-Flow CG-adaptive ({b.name})", x, residual, iterations, function_evaluations, converged, diverged, [""] * batch_size)


def bicg(b: Backend, lin, y, x0, check_progress: Callable, max_iter, poly_order: int) -> SolveResult or List[SolveResult]:
    """ Adapted from [BiCGstab for linear equations involving unsymmetric matrices with complex spectrum](https://dspace.library.uu.nl/bitstream/handle/1874/16827/sleijpen_93_bicgstab.pdf) """
    # Based on "BiCGstab(L) for linear equations involving unsymmetric matrices with complex spectrum" by Gerard L.G. Sleijpen
    y = b.to_float(y)
    x = b.copy(b.to_float(x0), only_mutable=True)
    batch_size = b.staticshape(y)[0]
    r0_tild = residual = y - b.linear(lin, x)
    iterations = b.zeros([batch_size], DType(int, 32))
    function_evaluations = b.ones([batch_size], DType(int, 32))
    residual_squared = b.sum(residual ** 2, -1, keepdims=True)
    continue_, converged, diverged = check_progress(iterations, residual_squared)
    rho_0 = b.ones([batch_size, 1])
    rho_1 = b.ones([batch_size, 1])
    omega = b.ones([batch_size, 1])
    alpha = b.zeros([batch_size, 1])
    u = b.zeros_like(x)
    r0_hat = [b.zeros(x0.shape)] * (poly_order + 1)
    u_hat = [b.zeros(x0.shape)] * (poly_order + 1)

    def loop_body(continue_, x, residual, iterations, function_evaluations, _converged, _diverged, rho_0, rho_1, omega, alpha, u, u_hat, r0_hat):
        tau = [[b.zeros((batch_size,))] * (poly_order + 1)] * (poly_order + 1)
        sigma = [b.zeros((batch_size,))] * (poly_order + 1)
        gamma = [b.zeros((batch_size,))] * (poly_order + 1)
        gamma_p = [b.zeros((batch_size,))] * (poly_order + 1)
        gamma_pp = [b.zeros((batch_size,))] * (poly_order + 1)
        continue_1 = b.to_int32(continue_)
        iterations += continue_1
        u_hat[0] = u
        r0_hat[0] = residual
        rho_0 = -omega * rho_0
        # --- Bi-CG part ---
        for j in range(0, poly_order):
            rho_1 = b.sum(r0_hat[j] * r0_tild, axis=-1, keepdims=True)
            beta = alpha * rho_1 / rho_0
            rho_0 = rho_1
            for i in range(0, j + 1):
                u_hat[i] = beta * u_hat[i]
                u_hat[i] = r0_hat[i] - u_hat[i]
            u_hat[j + 1] = b.linear(lin, u_hat[j]); function_evaluations += continue_1
            gamma_coeff = b.sum(u_hat[j + 1] * r0_tild, axis=-1, keepdims=True)
            alpha = rho_0 / gamma_coeff
            for i in range(0, j + 1):
                r0_hat[i] = r0_hat[i] - alpha * u_hat[i + 1]
            r0_hat[j + 1] = b.linear(lin, r0_hat[j]); function_evaluations += continue_1
            x = x + alpha * u_hat[0]
        for j in range(1, poly_order + 1):
            for i in range(1, j):
                tau[i][j] = b.sum(r0_hat[j] * r0_hat[i], axis=-1, keepdims=True) / sigma[i]
                r0_hat[j] = r0_hat[j] - tau[i][j] * r0_hat[i]
            sigma[j] = b.sum(r0_hat[j] * r0_hat[j], axis=-1, keepdims=True)
            gamma_p[j] = b.sum(r0_hat[0] * r0_hat[j], axis=-1, keepdims=True) / sigma[j]
        # --- MR part ---
        omega = gamma[poly_order] = gamma_p[poly_order]
        for j in range(poly_order - 1, 0, -1):
            sumg = b.zeros_like(tau[0][0])
            for i in range(j + 1, poly_order + 1):
                sumg = sumg + tau[j][i] * gamma[i]
            gamma[j] = gamma_p[j] - sumg
        for j in range(1, poly_order):
            sumg = b.zeros_like(tau[0][0])
            for i in range(j + 1, poly_order):
                sumg = sumg + tau[j][i] * gamma[i + 1]
            gamma_pp[j] = gamma[j + 1] + sumg
        # --- Update ---
        x = x + gamma[1] * r0_hat[0]
        r0_hat[0] = r0_hat[0] - gamma_p[poly_order] * r0_hat[poly_order]
        u_hat[0] = u_hat[0] - gamma[poly_order] * u_hat[poly_order]
        for j in range(1, poly_order):
            u_hat[0] = u_hat[0] - gamma[j] * u_hat[j]
            x = x + gamma_pp[j] * r0_hat[j]
            r0_hat[0] = r0_hat[0] - gamma_p[j] * r0_hat[j]
        u = u_hat[0]
        residual = r0_hat[0]
        residual_squared = b.sum(residual ** 2, -1, keepdims=True)
        continue_, converged, diverged = check_progress(iterations, residual_squared)
        # ToDo multiply step_size by continue_1 to avoid NaN when iterating after convergence
        return continue_, x, residual, iterations, function_evaluations, converged, diverged, rho_0, rho_1, omega, alpha, u, u_hat, r0_hat

    _, x, residual, iterations, function_evaluations, converged, diverged, rho_0, rho_1, omega, alpha, u, u_hat, r0_hat = b.while_loop(loop_body, (continue_, x, residual, iterations, function_evaluations, converged, diverged, rho_0, rho_1, omega, alpha, u, u_hat, r0_hat), _max_iter(max_iter))
    return SolveResult(f"Φ-Flow biCG-stab({poly_order}) ({b.name})", x, residual, iterations, function_evaluations, converged, diverged, [""] * batch_size)


def incomplete_lu_dense(b: 'Backend', matrix, iterations: int, safe: bool):
    """

    Args:
        b: `Backend`
        matrix: Square matrix of Shape (batch_size, rows, cols, channels)
        iterations: Number of fixed-point iterations to perform.
        safe: Avoid NaN when the rank deficiency of `matrix` is 2 or higher.
            For a rank deficiency of 1, the fixed-point algorithm will still converge without NaNs and all values of L and U are uniquely determined.
            If enabled, the algorithm is slightly slower.
            Rank deficiencies of 1 occur frequently in periodic settings but higher ones are rare.

    Returns:
        L: lower triangular matrix with ones on the diagonal
        U: upper triangular matrix
    """
    row, col = np.indices(b.staticshape(matrix)[1:-1])
    is_lower = np.expand_dims(row > col, -1)
    is_upper = np.expand_dims(row < col, -1)
    is_diagonal = np.expand_dims(row == col, -1)
    # # --- Initialize U as the diagonal of A, then compute off-diagonal of L ---
    lower = matrix / b.expand_dims(b.get_diagonal(matrix), 1)  # Since U=diag(A), L can be computed by a simple division
    lu = matrix * is_diagonal + lower * is_lower  # combine lower + diag(A) + 0
    # --- Fixed-point iterations ---
    for sweep in range(iterations):
        diag = b.expand_dims(b.get_diagonal(lu), 1)  # should never contain 0
        sum_l_u = b.einsum('bikc,bkjc->bijc', lu * is_lower, lu * is_upper)
        l = (matrix - sum_l_u) / diag if not safe else b.divide_no_nan(matrix - sum_l_u, diag)
        lu = b.where(is_lower, l, matrix - sum_l_u)
    # --- Assemble L=lower+unit_diagonal and U. ---
    return lu * is_lower + is_diagonal, lu * ~is_lower


def incomplete_lu_coo(b: 'Backend', indices, values, shape: Tuple[int, int], iterations: int, safe: bool):
    """
    Based on *Parallel Approximate LU Factorizations for Sparse Matrices* by T.K. Huckle, https://www5.in.tum.de/persons/huckle/it_ilu.pdf.

    Every matrix in the batch must explicitly store the full diagonal.
    There should not be any zeros on the diagonal, else the LU initialization fails.

    Args:
        b: `Backend`
        indices: Row & column indices of stored entries as `numpy.ndarray` of shape (batch_size, nnz, 2).
        values: Backend-compatible values tensor of shape (batch_size, nnz, channels)
        shape: Dense shape of matrix
        iterations: Number of fixed-point iterations to perform.
        safe: Avoid NaN when the rank deficiency of `matrix` is 2 or higher.
            For a rank deficiency of 1, the fixed-point algorithm will still converge without NaNs.
            If enabled, the algorithm is slightly slower.
            Rank deficiencies of 1 occur frequently in periodic settings but higher ones are rare.

    Returns:
        L: tuple `(indices, values)` where `indices` is a NumPy array and values is backend-specific
        U: tuple `(indices, values)` where `indices` is a NumPy array and values is backend-specific
    """
    assert isinstance(indices, np.ndarray), "incomplete_lu_coo indices must be a NumPy array"
    row, col = indices[..., 0], indices[..., 1]
    batch_size, nnz, channels = b.staticshape(values)
    rows, cols = shape
    assert rows == cols, "incomplete_lu_coo only implemented for square matrices"
    is_lower = np.expand_dims(row > col, -1)
    diagonal_indices = np.expand_dims(get_lower_diagonal_indices(row, col, shape), -1)  # indices of corresponding values that lie on the diagonal
    is_diagonal = np.expand_dims(row == col, -1)
    mm_above, mm_left, mm_is_valid = strict_lu_mm_pattern_coo_batched(row, col, rows, cols)
    mm_above = np.expand_dims(mm_above, -1)
    mm_left = np.expand_dims(mm_left, -1)
    mm_is_valid = np.expand_dims(mm_is_valid, -1)
    # --- Initialize U as the diagonal of A, then compute off-diagonal of L ---
    lower = values / b.batched_gather_nd(values, diagonal_indices)  # Since U=diag(A), L can be computed by a simple division
    lu = values * is_diagonal + lower * is_lower  # combine lower + diag(A) + 0
    # --- Fixed-point iterations ---
    for sweep in range(iterations):
        diag = b.batched_gather_nd(lu, diagonal_indices)  # should never contain 0
        sum_l_u = b.einsum('bnkc,bnkc->bnc', b.batched_gather_nd(lu, mm_above) * mm_is_valid, b.batched_gather_nd(lu, mm_left))
        l = (values - sum_l_u) / diag if not safe else b.divide_no_nan(values - sum_l_u, diag)
        lu = b.where(is_lower, l, values - sum_l_u)
    # --- Assemble L=lower+unit_diagonal and U. If nnz varies along batch, keep the full sparsity pattern ---
    u_values = b.where(~is_lower, lu, 0)
    belongs_to_lower = (is_lower | is_diagonal)
    l_values = b.where(is_lower, lu, b.cast(is_diagonal, b.dtype(values)))
    u_mask_indices_b, u_mask_indices = np.where(~is_lower[..., 0])
    _, u_nnz = np.unique(u_mask_indices_b, return_counts=True)
    if np.all(u_nnz == u_nnz[0]):  # nnz for lower/upper does not vary along batch
        u_mask_indices = np.reshape(u_mask_indices, (batch_size, -1))
        u_values = b.batched_gather_nd(u_values, np.expand_dims(u_mask_indices, -1))
        u_indices = np.stack([indices[b, u_mask_indices[b], :] for b in range(batch_size)])
        _, l_mask_indices = np.where(belongs_to_lower[..., 0])
        l_mask_indices = np.reshape(l_mask_indices, (batch_size, -1))
        l_values = b.batched_gather_nd(l_values, np.expand_dims(l_mask_indices, -1))
        l_indices = np.stack([indices[b, l_mask_indices[b], :] for b in range(batch_size)])
        return (l_indices, l_values), (u_indices, u_values)
    else:  # Keep all indices since the number in lower/upper varies along the batch
        return (indices, l_values), (indices, u_values)


def strict_lu_mm_pattern_coo(row: np.ndarray, col: np.ndarray, rows, cols):
    """
    For each stored entry e at (row, col), finds the matching entries directly above and directly left of e, such that left.col == above.row.

    This is useful for multiplying a lower triangular and upper triangular matrix given the sparsity pattern but excluding the diagonals.
    The matrix multiplication then is given by
    >>> einsum('nk,nk->n', stored_lower[above_entries] * is_valid_entry, stored_upper[left_entries])

    Returns:
        above_entries: (max_num, nnz) Stored indices of matched elements above any entry.
        left_entries: (max_num, nnz) Stored indices of matched elements to the left of any entry.
        is_valid_entry: (max_num, nnz) Mask of valid indices. Invalid indices are undefined but lie inside the array to prevent index errors.
    """
    entries, = row.shape
    # --- Compress rows and cols ---
    lower_entries_by_row = compress_strict_lower_triangular_rows(row, col, rows)  # entry indices by row, -1 for non-existent entries
    upper_entries_by_col = compress_strict_lower_triangular_rows(col, row, cols)
    # --- Find above and left entries ---
    same_row_entries = lower_entries_by_row[:, row]  # (row entries, entries). Currently, contains valid values for invalid references
    left = np.where(col[same_row_entries] < col, same_row_entries, -1)  # (max_left, nnz)  all entries with col_e==col, row_e < row
    same_col_entries = upper_entries_by_col[:, col]
    above = np.where(row[same_col_entries] < row, same_col_entries, -1)  # (max_above, nnz)
    # --- for each entry, match left and above where left.col == above.row ---
    half_density = max(len(lower_entries_by_row), len(upper_entries_by_col))
    above_entries = np.zeros([entries, half_density], dtype=int)
    left_entries = np.zeros([entries, half_density], dtype=int)
    is_valid_entry = np.zeros([entries, half_density])
    k = np.zeros(entries, dtype=int)
    for r in range(len(above)):
        for c in range(len(left)):
            match = (col[left[c]] == row[above[r]]) & (above[r] != -1)
            where_match = np.where(match)
            k_where_match = k[where_match]
            above_entries[where_match, k_where_match] = above[r][where_match]
            left_entries[where_match, k_where_match] = left[c][where_match]
            is_valid_entry[where_match, k_where_match] = 1
            k += match
    return above_entries, left_entries, is_valid_entry


def compress_strict_lower_triangular_rows(row, col, rows):
    is_lower = row > col
    below_diagonal = np.where(is_lower)
    row_lower = row[below_diagonal]
    num_in_row = get_index_in_row(row_lower, col[below_diagonal])
    lower_entries_by_row = np.zeros((np.max(num_in_row)+1, rows), dtype=row.dtype) - 1
    lower_entries_by_row[num_in_row, row_lower] = below_diagonal
    return lower_entries_by_row


def strict_lu_mm_pattern_coo_batched(row, col, rows, cols):
    results = [strict_lu_mm_pattern_coo(row[b], col[b], rows, cols) for b in range(row.shape[0])]
    result = [np.stack(v) for v in zip(*results)]
    return result


def get_index_in_row(row: np.ndarray, col: np.ndarray):
    """ How many entries are to the left of a given entry but in the same row, i.e. the how manieth index this is per row. """
    perm = np.argsort(col)
    compressed_col_index = cumcount(row[perm])[inv_perm(perm)]
    return compressed_col_index


def inv_perm(perm):
    """ Returns the permutation necessary to undo a sort given the argsort array. """
    u = np.empty(perm.size, dtype=np.int64)
    u[perm] = np.arange(perm.size)
    return u


def cumcount(a):
    """ Based on https://stackoverflow.com/questions/40602269/how-to-use-numpy-to-get-the-cumulative-count-by-unique-values-in-linear-time """
    def dfill(a):
        """ Returns the positions where the array changes and repeats that index position until the next change. """
        b = np.concatenate([[0], np.where(a[:-1] != a[1:])[0] + 1, [a.size]])
        return np.arange(a.size)[b[:-1]].repeat(np.diff(b))
    perm = a.argsort(kind='mergesort')
    inv = inv_perm(perm)
    return (np.arange(a.size) - dfill(a[perm]))[inv]


def cumcount2(l):  # slightly slower than cumcount
    a = np.unique(l, return_counts=True)[1]
    idx = a.cumsum()
    id_arr = np.ones(idx[-1], dtype=int)
    id_arr[0] = 0
    id_arr[idx[:-1]] = -a[:-1] + 1
    rng = id_arr.cumsum()
    return rng[inv_perm(np.argsort(l))]


def get_transposed_indices(row, col, shape):
    linear = np.ravel_multi_index((row, col), shape)
    linear_transposed = np.ravel_multi_index((col, row), shape)
    has_transpose = np.stack([np.isin(linear[b], linear_transposed[b]) for b in range(row.shape[0])])
    perm = np.argsort(linear)
    transposed = np.stack([np.searchsorted(linear[b], linear_transposed[b], sorter=perm[b]) for b in range(row.shape[0])])
    transposed = np.minimum(transposed, len(row) - 1)
    return has_transpose, transposed


def get_lower_diagonal_indices(row, col, shape):
    linear = np.ravel_multi_index((row, col), shape)
    j = np.minimum(row, col)
    diagonal_indices = np.ravel_multi_index((j, j), shape)
    perm = np.argsort(linear)
    result = [perm[b, np.searchsorted(linear[b], diagonal_indices[b], sorter=perm[b])] for b in range(row.shape[0])]
    assert np.all([np.isin(diagonal_indices[b], linear[b]) for b in range(row.shape[0])]), "All diagonal elements must be present in sparse matrix."
    return np.stack(result)


def parallelize_dense_triangular_solve(b: Backend, matrix, lower_triangular=True):
    """

    Args:
        b:
        matrix: lower-triangular matrix

    Returns:

    """
    rows, cols = b.staticshape(matrix)
    # batch_size, rows, cols, channels = b.staticshape(matrix)
    xs = {}
    for row in range(rows):
        x = b.zeros((cols,))
        x[row] = 1
        for j in range(row):
            x -= xs[j] * matrix[row, j]
        if not lower_triangular:
            x /= matrix[row, row]
        xs[row] = x
    print(xs)
