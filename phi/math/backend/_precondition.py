from typing import Tuple

import numpy as np

from ._backend import Backend
from ._dtype import DType, to_numpy_dtype


def incomplete_lu_coo(b: 'Backend', indices, values, shape: Tuple[int, int], iterations: int):
    """
    Based on *Parallel Approximate LU Factorizations for Sparse Matrices* by T.K. Huckle, https://www5.in.tum.de/persons/huckle/it_ilu.pdf.

    Every matrix in the batch must explicitly store the full diagonal.
    There should not be any zeros on the diagonal, else the LU initialization fails.

    Args:
        b: `Backend`
        indices: Row & column indices of stored entries as `numpy.ndarray` of shape (batch_size, nnz, 2).
        values: Backend-compatible values tensor of shape (batch_size, nnz, channels)
        shape: Dense shape of matrix
        iterations: Number of sweeps to perform.

    Returns:
        lower: tuple (indices, values) where indices is a NumPy array and values is backend-specific
        upper: tuple (indices, values) where indices is a NumPy array and values is backend-specific
    """
    assert isinstance(indices, np.ndarray), "incomplete_lu_coo indices must be a NumPy array"
    row, col = indices[..., 0], indices[..., 1]
    batch_size, nnz, channels = b.staticshape(values)
    rows, cols = shape
    assert rows == cols, "incomplete_lu_coo only implemented for square matrices"
    is_lower = np.expand_dims(row > col, -1)
    index_in_row = get_index_in_row(row, col)
    index_in_row_ = np.stack([row, index_in_row], -1)
    max_entries_per_row = np.max(index_in_row)
    has_transpose, transposed_index = get_transposed_indices(row, col, shape)  # The corresponding index in the transposed pattern. If non-existent, points at any valid value
    transposed_index = np.expand_dims(transposed_index, -1)
    has_transpose = b.cast(np.expand_dims(has_transpose, -1), b.dtype(values))  # 0 or 1 depending on whether a transposed entry exists for a value
    diagonal_indices = np.expand_dims(get_lower_diagonal_indices(row, col, shape), -1)  # indices of corresponding values that lie on the diagonal
    l_u_compressed_zeros = b.zeros((batch_size, rows, max_entries_per_row + 1, channels))
    # --- Initialize U as the diagonal of A, then compute off-diagonal of L ---
    is_diagonal = np.expand_dims(row == col, -1)
    lower = values / b.batched_gather_nd(values, diagonal_indices)  # Since U=diag(A), L can be computed by a simple division
    lu = b.where(is_diagonal, values, b.where(is_lower, lower, 0))  # combine lower + diag(A) + 0
    # --- Fixed-point iterations ---
    for sweep in range(iterations):
        diag = b.batched_gather_nd(lu, diagonal_indices)  # should never contain 0
        l_u = lu * b.batched_gather_nd(lu, transposed_index) * has_transpose  # matches indices (like lu, values)
        # --- Temporarily densify indices by row for cumsum ---
        l_u_compressed = b.scatter(l_u_compressed_zeros, b.stack([row, index_in_row], -1), l_u, mode='add')
        sum_l_u = b.cumsum(l_u_compressed, -2)
        sum_l_u = b.batched_gather_nd(sum_l_u, index_in_row_)
        # --- update L and U in one matrix ---
        l = 1 / diag * (values - sum_l_u)
        u = values - sum_l_u
        lu = b.where(is_lower, l, u)
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


def get_index_in_row(row: np.ndarray, col: np.ndarray):
    """ How many entries are to the left of a given entry but in the same row, i.e. the how manieth index this is per row. """
    perm = np.argsort(col)
    compressed_col_index = [cumcount(row[b][perm[b]])[inv_perm(perm[b])] for b in range(row.shape[0])]
    return np.stack(compressed_col_index)


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
