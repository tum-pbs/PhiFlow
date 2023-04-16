from numbers import Number
from typing import Tuple, Any

import numpy as np

from ._dtype import DType, to_numpy_dtype
from ._backend import Backend


def neighbor_search(b: Backend,
                    positions,
                    max_dist,
                    tmp_pair_count: int = None,
                    pair_count: int = None,
                    table_len: int = None,
                    index_dtype: DType = DType(int, 32),
                    trim: bool = True,
                    default: Number = float('nan'),
                    cell_mode: str = 'grid',
                    pair_by: str = 'gather-search') -> tuple:
    """

    Args:
        b: `Backend`
        positions: Point locations of shape (batch, instances, vector)
        max_dist: Scalar float
        tmp_pair_count: Maximum number of total particle pairs, including ones in neighboring cells that are more than `max_dist` apart.
            This must be set when jit-compiling this function with Jax.
        pair_count: Length of output lists.
            This must be set when jit-compiling this function with Jax.
        table_len: Maximum lookup table size. For `cell_mode='grid'`, this equals the maximum number of virtual cells.
            This must be set when jit-compiling this function with Jax.
        index_dtype: Either int32 or int64
        trim: Whether to only include pairs for particles in range. If `False`, returns a possibly larger list with invalid deltas represented as `default´.
            If `pair_count` is specified the result list may be of that size and fill unused values with `default` anyway.
        default: Value to return for particles that are further apart than `max_dist` or otherwise invalid values of `differences`.
        cell_mode: Only `'box'` currently supported.
        pair_by: Method to use to gather particle pairs into a pair-list.

    Returns:
        source_indices: (pair_count,)
        target_indices: (pair_count,)
        differences: (pair_count, ndims)
    """
    assert b.ndims(max_dist) == 0, f"Only scalar max_distance currently supported for sparse distances"
    positions = b.to_float(b.as_tensor(positions))
    max_dist = b.to_float(max_dist)
    num_part, d = b.staticshape(positions)
    cells, perm, neighbor_cells, structure = build_cells(b, positions, max_dist, cell_mode, table_len=table_len)
    linear_indices = b.range(num_part, dtype=index_dtype)
    particle_ids = b.gather(linear_indices, perm, 0)
    positions = b.gather(positions, perm, 0)
    # --- Find particles in neighbor cells for each particle ---
    num_neighbors_by_direction = structure.get_num_elements_in_cell(neighbor_cells)
    num_potential_neighbors = b.sum(num_neighbors_by_direction, 0)
    num_required_tmp_pairs = b.sum(num_potential_neighbors)
    long_linear_indices = b.range(tmp_pair_count or num_required_tmp_pairs, dtype=index_dtype)
    if pair_by == 'scatter':
        raise NotImplementedError
        # to_id = b.zeros(sum(num_potential_neighbors), dtype=index_dtype)  # ToDo bound this by some constant (jit)
        # offsets = structure.get_first_element_by_cell(cells)
        # for neighbor_cell_ids in neighbor_cells:
        #     to_id = b.scatter(to_id, offsets)  # ToDo either scatter (current) or gather (other viewpoint)
        #     offsets += ...
    elif pair_by == 'gather-search':
        neighbor_partitions = b.cumsum(num_neighbors_by_direction, 0)
        # first_in_cell = b.repeat(structure.get_first_element_by_cell(cells), num_potential_neighbors, 0)
        small_index = b.repeat(linear_indices, num_potential_neighbors, 0, new_length=tmp_pair_count)
        nb_part_idx_in_cell = long_linear_indices - b.repeat(b.cumsum(num_potential_neighbors, 0) - num_potential_neighbors, num_potential_neighbors, 0, new_length=tmp_pair_count)
        neighbor_partitions = b.repeat(b.transpose(neighbor_partitions, (1, 0)), num_potential_neighbors, 0, new_length=tmp_pair_count)
        direction = b.searchsorted(neighbor_partitions, nb_part_idx_in_cell[:, None], 'right')[:, 0]
        neighbor_cell = b.gather_by_component_indices(neighbor_cells, direction, small_index)
        relative_particle_idx = nb_part_idx_in_cell - b.batched_gather_1d(neighbor_partitions, direction[:, None] - 1)[:, 0]
        to_idx = structure.get_first_element_by_cell(neighbor_cell) + relative_particle_idx
    else:
        raise ValueError(pair_by)
    # --- Lookup positions and compute distances ---
    from_id = b.repeat(particle_ids, num_potential_neighbors, 0, new_length=tmp_pair_count)
    to_id = b.gather(particle_ids, to_idx, 0)
    from_positions = b.repeat(positions, num_potential_neighbors, 0, new_length=tmp_pair_count)
    to_positions = b.gather(positions, to_idx, 0)
    dx = to_positions - from_positions
    # --- Filter by max_dist ---
    dist = b.sqrt(b.sum(dx ** 2, -1))
    in_range = dist < max_dist
    valid = in_range & (long_linear_indices < num_required_tmp_pairs)
    if trim and (tmp_pair_count is None or (pair_count is not None and pair_count < tmp_pair_count)):
        pair_count = pair_count or tmp_pair_count
        dx = b.boolean_mask(dx, valid, new_length=pair_count, fill_value=default)
        from_id = b.boolean_mask(from_id, valid, new_length=pair_count, fill_value=-1)
        to_id = b.boolean_mask(to_id, valid, new_length=pair_count, fill_value=-1)
    else:
        dx = b.where(valid[:, None], dx, default)
        from_id = b.where(valid, from_id, -1)
        to_id = b.where(valid, from_id, -1)
    return from_id, to_id, dx


def build_cells(b: Backend, positions, min_cell_size, mode: str, table_len: int = None) -> Tuple[Any, Any, Any, 'IndexingStructure']:
    assert mode in ['grid', 'tree'], mode  # , 'hexagonal-grid'
    num_part, d = b.staticshape(positions)
    lower = b.min(positions, 0)
    upper = b.max(positions, 0) + 1e-5
    extent = b.maximum(upper - lower, 1e-15)
    if mode == 'grid':
        resolution = b.maximum(1, b.to_int32(extent / min_cell_size))  # ToDo decrease resolution if larger than table_len
        cell_indices = b.to_int32((positions - lower) / extent * b.to_float(resolution))
        cell_ids = b.ravel_multi_index(cell_indices, resolution)
        perm = b.argsort(cell_ids)
        cell_indices = b.gather(cell_indices, perm, 0)
        cell_ids = b.gather(cell_ids, perm, 0)
        num_virtual_cells = b.prod(resolution, 0)
        idx_by_cell = b.searchsorted(cell_ids, b.range(table_len or num_virtual_cells), 'left')
        len_by_cell = b.bincount(cell_ids, None, bins=num_part)
        neighbor_offsets = b.as_tensor(np.reshape(np.stack(np.meshgrid(*[(-1, 0, 1)] * d, indexing='ij'), -1), (-1, 1, d)))
        neighbor_ids = b.ravel_multi_index(cell_indices + neighbor_offsets, resolution, mode=-1)
        return cell_ids, perm, neighbor_ids, CellArray(b, idx_by_cell, len_by_cell)
    raise NotImplementedError(mode)


class IndexingStructure:

    def get_first_element_by_cell(self, cell):
        raise NotImplementedError

    def get_num_elements_in_cell(self, cell):
        raise NotImplementedError


class CellArray(IndexingStructure):
    """Explicitly lists all cells, even unused ones."""

    def __init__(self, b: Backend, first_idx_by_cell, num_by_cell):
        self._b = b
        self._first_idx_by_cell = first_idx_by_cell
        self._num_by_cell = num_by_cell

    def get_first_element_by_cell(self, cell):
        return self._b.gather(self._first_idx_by_cell, cell, 0)

    def get_num_elements_in_cell(self, cell):
        return self._b.where(cell < 0, 0, self._b.gather(self._num_by_cell, cell, 0))


class CellTree(IndexingStructure):
    ...


class CellHashMap(IndexingStructure):
    ...


# ToDo hierarchy of cells if too many cells to list all explicitly. Sorting by lowest index must also sort for higher indices


def pairwise_distances_sklearn(self: Backend, positions, max_dist, index_dtype=DType(int, 32)):
    from sklearn import neighbors
    batch_size, point_count, _vec_count = self.staticshape(positions)
    positions_np_batched = self.numpy(positions)
    result = []
    for i in range(batch_size):
        tree = neighbors.KDTree(positions_np_batched[i])
        radius = float(max_dist) if len(self.staticshape(max_dist)) == 0 else max_dist[i]
        nested_neighbors = tree.query_radius(positions_np_batched[i], r=radius)  # ndarray[ndarray]
        column_indices = np.concatenate(nested_neighbors).astype(to_numpy_dtype(index_dtype))  # flattened_neighbors
        neighbor_counts = [len(nlist) for nlist in nested_neighbors]
        row_pointers = np.concatenate([[0], np.cumsum(neighbor_counts)]).astype(to_numpy_dtype(index_dtype))
        pos_neighbors = self.gather(positions[i], column_indices, 0)
        pos_self = self.repeat(positions[i], neighbor_counts, axis=0)
        values = pos_neighbors - pos_self
        result.append((column_indices, row_pointers, values))
        # sparse_matrix = self.csr_matrix(column_indices, row_pointers, values, (point_count, point_count))
        # sparse_matrix.eliminate_zeros()  # setdiag(0) keeps zero entries
    return result
