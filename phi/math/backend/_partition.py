from numbers import Number
from typing import Tuple, Any, Union, Optional

import numpy as np

from ._dtype import DType, to_numpy_dtype
from ._backend import Backend, choose_backend, TensorType


def find_neighbors_semi_sparse(positions,
                               cutoff,
                               domain: Optional[Tuple[TensorType, TensorType]] = None,
                               periodic: Union[bool, Tuple[bool, ...]] = False,
                               pair_count: int = None,
                               max_occupancy: int = None,
                               index_dtype: DType = DType(int, 32),
                               default: Number = float('nan'),
                               check_sizes_sufficient=True) -> tuple:
    """
    Neighbor search with JIT support.
    This implementation is loosely based on the neighbor_list function from Jax-MD.
    It pads each cell to a common occupancy (`max_occupancy`), filling in empty slots with ´-1`.

    Args:
        positions: Point locations of shape (batch, instances, vector)
        cutoff: Scalar float or 1D tensor
        domain: (Optional) Lower and upper corner of domain.
        periodic: Whether domain boundaries are periodic.
        pair_count: Length of output lists.
            This must be set when jit-compiling this function with Jax.
        max_occupancy: Maximum number of elements in any cell.
            Must be specified when tracing this function.
        index_dtype: Either int32 or int64.
        default: Value to return for particles that are further apart than `max_dist` or otherwise invalid values of `differences`.
        check_sizes_sufficient: If `True`, computes `required_pair_count` and `required_max_occupancy`.

    Returns:
        source_indices: (pair_count,)
        target_indices: (pair_count,)
        differences: (pair_count, ndims)
        required_pair_count: If `check_sizes_sufficient`, returns the number of pairs.
            If `max_occupancy` was set too small, `required_pair_count` cannot be correctly determined and `-1` will be returned instead.
        required_max_occupancy: If `check_sizes_sufficient` or `max_occupancy` is not specified, returns the maximum cell occupancy number.
    """
    b = choose_backend(positions)
    if not b.is_available(positions):
        assert pair_count is not None, f"pair_count must be provided when tracing find_neighbors"
        assert max_occupancy is not None, f"pair_count must be provided when tracing find_neighbors"
    positions = b.to_float(b.as_tensor(positions))
    n, d = b.staticshape(positions)
    min_cell_size = cutoff
    b_ = choose_backend(min_cell_size, *domain or ())
    periodic = [periodic] * d if np.ndim(periodic) == 0 else tuple(periodic)
    if domain is None:
        domain = b_.min(positions, 0), b_.max(positions, 0) + 1e-5
    else:
        domain = (b_.as_tensor(domain[0]), b_.as_tensor(domain[1]))
    domain_size = b_.maximum(domain[1] - domain[0], min_cell_size)
    resolution = b_.maximum(1, b_.to_int32(b_.ceil(domain_size / min_cell_size)))
    cell_count = b_.prod(resolution)
    extent = b.as_tensor(min_cell_size * resolution)
    cell_indices = b.to_int32((positions - b.to_float(domain[0])) / extent * b.to_float(resolution))
    cell_ids = b.ravel_multi_index(cell_indices, resolution)  # hash in Jax-MD
    # --- sort by cell id ---
    perm = b.argsort(cell_ids)
    cell_ids = b.gather(cell_ids, perm, 0)
    positions = b.gather(positions, perm, 0)
    particle_ids = b.gather(b.range(n, dtype=index_dtype), perm, 0)
    # --- check max occupancy ---
    if max_occupancy is None or check_sizes_sufficient:
        required_max_occupancy = b.max(b.bincount(cell_ids, None, bins=cell_count, x_sorted=True))
        if max_occupancy is None:
            assert b.is_available(positions), f"max_occupancy must be specified when tracing"
        max_occupancy = max_occupancy or required_max_occupancy
    else:
        required_max_occupancy = None
    # --- scatter particle ids into cell contents ---
    idx_in_cell_contents = b.stack([cell_ids, b.range(n) % max_occupancy], -1)
    cell_contents = b.scatter_nd_scalar(b.zeros((cell_count, max_occupancy), index_dtype) - 1, idx_in_cell_contents, particle_ids, 'update')
    # --- accumulate neighbor cell contents ---
    neighbor_offsets = np.reshape(np.stack(np.meshgrid(*[(-1, 0, 1)] * d, indexing='ij'), -1), (-1, d))
    neighbor_count = b.staticshape(neighbor_offsets)[0]
    neighbor_contents = b.stack([shift(d, cell_contents, offset, resolution, periodic) for offset in neighbor_offsets], -2)  # (cell_count, neighbor_count, max_occupancy)
    neighbor_contents = b.tile(neighbor_contents[:, None, :, :], (1, max_occupancy, 1, 1))
    # ---  compute deltas, prune by distance ---
    neighbor_idx = b.scatter_nd(b.zeros((n + 1, neighbor_count * max_occupancy), index_dtype), indices=b.reshape(cell_contents, (-1, 1)) + 1, values=b.reshape(neighbor_contents, (-1, neighbor_count * max_occupancy)), mode='update')[1:]  # copy blocks of (neighbor_count, max_occupancy):  Index (cell_count * max_occupancy)  Channels (neighbors * max_occupancy)
    neighbor_pos = b.batched_gather_nd(positions[None, :, :], b.maximum(0, neighbor_idx[:, :, None]))
    dx = positions[:, None, :] - neighbor_pos
    mask = (b.sum(dx ** 2, -1) < cutoff ** 2) & (neighbor_idx >= 0)
    required_pair_count = b.where(required_max_occupancy > max_occupancy, -1, b.sum(mask)) if check_sizes_sufficient else None
    entries = b.nonzero(mask, length=pair_count, fill_value=-1)
    from_id = entries[:, 0]
    to_id = b.gather_nd(neighbor_idx[:, :, None], b.maximum(0, entries))[:, 0]
    dx = b.where(from_id[:, None] >= 0, b.gather_nd(dx, entries), default)
    return from_id, to_id, dx, required_pair_count, required_max_occupancy


def shift(d: int, array, offset, resolution, periodic: Tuple[bool, ...], edge: int = -1):
    b = choose_backend(array)
    array = b.reshape(array, (*resolution, *b.staticshape(array)[1:]))
    for i, shift in enumerate(offset):
        if shift > 0:
            slices1 = tuple([slice(-1, None) if j == i else slice(None) for j in range(d)])
            slices2 = tuple([slice(0, -1) if j == i else slice(None) for j in range(d)])
            edge_values = array[slices1] if periodic[i] else b.zeros_like(array[slices1]) + edge
            array = b.concat([edge_values, array[slices2]], i)
        elif shift < 0:
            slices1 = tuple([slice(1, None) if j == i else slice(None) for j in range(d)])
            slices2 = tuple([slice(0, 1) if j == i else slice(None) for j in range(d)])
            edge_values = array[slices2] if periodic[i] else b.zeros_like(array[slices2]) + edge
            array = b.concat([array[slices1], edge_values], i)
    return b.reshape(array, (-1, *b.staticshape(array)[d:]))


def find_neighbors_sparse(positions,
                          cutoff,
                          domain: Optional[Tuple[TensorType, TensorType]] = None,
                          periodic: Union[bool, Tuple[bool, ...]] = False,
                          pair_count: int = None,
                          index_dtype: DType = DType(int, 32),
                          trim: bool = True,
                          default: Number = float('nan'),
                          pair_by: str = 'gather-search') -> tuple:
    """
    Neighbor search with JIT support.
    This implementation uses pointers to reference cell contents instead of padding each cell to a common occupancy.

    Args:
        positions: Point locations of shape (batch, instances, vector)
        cutoff: Scalar float or 1D tensor
        domain: (Optional) Lower and upper corner of domain.
        periodic: Whether domain boundaries are periodic.
        pair_count: Length of output lists.
            This must be set when jit-compiling this function with Jax.
        index_dtype: Either int32 or int64.
        trim: Whether to only include pairs for particles in range. If `False`, returns a possibly larger list with invalid deltas represented as `default´.
            If `pair_count` is specified the result list may be of that size and fill unused values with `default` anyway.
        default: Value to return for particles that are further apart than `max_dist` or otherwise invalid values of `differences`.
        pair_by: Method to use to gather particle pairs into a pair-list.

    Returns:
        source_indices: (pair_count,)
        target_indices: (pair_count,)
        differences: (pair_count, #dims)
    """
    b = choose_backend(positions)
    if not b.is_available(positions):
        assert pair_count is not None, f"pair_count must be provided when tracing find_neighbors"
    positions = b.to_float(b.as_tensor(positions))
    n, d = b.staticshape(positions)
    cells, perm, neighbor_cells, cell_size, structure = build_cells(positions, cutoff, domain, periodic)
    b_ = choose_backend(cell_size)
    tmp_pair_count = b_.to_int32(b_.ceil(pair_count * b_.prod(3 * cell_size) / (_sphere_volume(cutoff, d)))) if pair_count is not None else None  # approximate over-counting from including out-of-range particles in neighboring cells
    linear_indices = b.range(n, dtype=index_dtype)
    particle_ids = b.gather(linear_indices, perm, 0)
    positions = b.gather(positions, perm, 0)
    # --- Find particles in neighbor cells for each particle ---
    num_neighbors_by_direction = structure.get_num_elements_in_cell(neighbor_cells)
    num_potential_neighbors = b.sum(num_neighbors_by_direction, 0)
    num_required_tmp_pairs = b.sum(num_potential_neighbors)
    pair_indices = b.range(tmp_pair_count or num_required_tmp_pairs, dtype=index_dtype)
    if pair_by == 'scatter':
        raise NotImplementedError
        # to_id = b.zeros(sum(num_potential_neighbors), dtype=index_dtype)  # ToDo bound this by some constant (jit)
        # offsets = structure.get_first_element_by_cell(cells)
        # for neighbor_cell_ids in neighbor_cells:
        #     to_id = b.scatter(to_id, offsets)  # ToDo either scatter (current) or gather (other viewpoint)
        #     offsets += ...
    elif pair_by == 'gather-search':
        neighbor_partitions = b.cumsum(num_neighbors_by_direction, 0)
        small_index = b.repeat(linear_indices, num_potential_neighbors, 0, new_length=tmp_pair_count)
        nb_part_idx_in_cell = pair_indices - b.repeat(b.cumsum(num_potential_neighbors, 0) - num_potential_neighbors, num_potential_neighbors, 0, new_length=tmp_pair_count)
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
    # --- Filter by cutoff ---
    dist = b.sqrt(b.sum(dx ** 2, -1))
    in_range = dist < cutoff
    valid = in_range & (pair_indices < num_required_tmp_pairs)
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


def build_cells(positions,
                min_cell_size,
                domain: Optional[Tuple[TensorType, TensorType]],
                periodic: Union[bool, Tuple[bool, ...]]) -> Tuple[TensorType, TensorType, TensorType, TensorType, 'IndexingStructure']:
    b = choose_backend(positions)
    b_ = choose_backend(min_cell_size, *domain or ())
    _, d = b.staticshape(positions)
    periodic = [periodic] * d if np.ndim(periodic) == 0 else tuple(periodic)
    if domain is None:
        domain = b_.min(positions, 0), b.max(positions, 0) + 1e-5
    domain_size = b_.maximum(domain[1] - domain[0], min_cell_size)
    resolution = b_.maximum(1, b_.to_int32(b_.ceil(domain_size / min_cell_size)))
    cell_count = b_.prod(resolution)
    extent = min_cell_size * resolution
    cell_size = extent / resolution
    cell_indices = b.to_int32((positions - domain[0]) / extent * b.to_float(resolution))
    cell_ids = b.ravel_multi_index(cell_indices, resolution)
    perm = b.argsort(cell_ids)
    cell_indices = b.gather(cell_indices, perm, 0)
    cell_ids = b.gather(cell_ids, perm, 0)
    idx_by_cell = b.searchsorted(cell_ids, b.range(cell_count), 'left')
    occupancy = b.bincount(cell_ids, None, bins=cell_count, x_sorted=True)
    neighbor_offsets = b.as_tensor(np.reshape(np.stack(np.meshgrid(*[(-1, 0, 1)] * d, indexing='ij'), -1), (-1, 1, d)))
    if any(periodic):
        assert all(periodic), f"Only fully periodic or fully non-periodic boundaries currently supported"
        neighbor_ids = b.ravel_multi_index(cell_indices + neighbor_offsets, resolution, mode='periodic')
    else:
        neighbor_ids = b.ravel_multi_index(cell_indices + neighbor_offsets, resolution, mode=-1)
    return cell_ids, perm, neighbor_ids, cell_size, CellArray(b, idx_by_cell, occupancy)


class IndexingStructure:

    def get_first_element_by_cell(self, cell):
        raise NotImplementedError

    def get_num_elements_in_cell(self, cell):
        raise NotImplementedError


class CellArray(IndexingStructure):
    """Explicitly lists all cells, even unused ones."""

    def __init__(self, b: Backend, first_idx_by_cell, occupancy_by_cell):
        self._b = b
        self._first_idx_by_cell = first_idx_by_cell
        self._num_by_cell = occupancy_by_cell

    def get_first_element_by_cell(self, cell):
        return self._b.gather(self._first_idx_by_cell, cell, 0)

    def get_num_elements_in_cell(self, cell):
        return self._b.where(cell < 0, 0, self._b.gather(self._num_by_cell, cell, 0))


class CellTree(IndexingStructure):
    ...


def _sphere_volume(radius, d: int):
    if d == 1:
        return 2 * radius
    elif d == 2:
        return np.pi * radius ** 2
    elif d == 3:
        return 4 / 3 * np.pi * radius ** 3
    raise NotImplementedError(f"d={d}")


# ToDo hierarchy of cells if too many cells to list all explicitly. Sorting by lowest index must also sort for higher indices


def find_neighbors_sklearn(positions: TensorType,
                           cutoff: Union[float, TensorType],
                           index_dtype=DType(int, 32)):
    from sklearn.neighbors import KDTree
    b = choose_backend(positions)
    np_positions = b.numpy(positions)
    tree = KDTree(np_positions)
    nested_neighbors = tree.query_radius(np_positions, r=cutoff)  # ndarray[ndarray]
    col = np.concatenate(nested_neighbors).astype(to_numpy_dtype(index_dtype), copy=False)  # flattened_neighbors
    neighbor_counts = [len(nlist) for nlist in nested_neighbors]
    row_pointers = np.concatenate([[0], np.cumsum(neighbor_counts)]).astype(to_numpy_dtype(index_dtype), copy=False)
    pos_neighbors = b.gather(positions, col, 0)
    pos_self = b.repeat(positions, neighbor_counts, axis=0)
    return col, row_pointers, pos_neighbors - pos_self


def find_neighbors_matscipy(positions: TensorType,
                            cutoff: Union[float, TensorType],
                            domain: Optional[Tuple[TensorType, TensorType]] = None,
                            periodic: Union[bool, Tuple[bool, ...]] = False,
                            index_dtype=DType(int, 32)):
    """
    Args:
        positions: (points, d) where 0 < d <= 3.
        cutoff: Global cutoff value.
        domain: Lower and upper box extent as a 2-`tuple` `(lower, upper)`
        periodic: Single `bool` or `tuple` specifying periodicity by dimension.
        index_dtype: Either int32 or int64.

    Returns:
        row: Row index (self / distance from)
        col: Column index (other / distance to)
        delta: Differentiable position differences (col - row)
    """
    from matscipy.neighbours import neighbour_list
    b = choose_backend(positions)
    pos_count, d = b.staticshape(positions)
    np_positions = b.numpy(positions)
    cutoff = float(cutoff)  # if b.ndims(cutoff) == 0 else b.numpy(cutoff).tolist()  # interpreted as overlap sphere radius if a list, else the cutoff
    max_cutoff = cutoff if isinstance(cutoff, float) else np.max(cutoff)
    periodic = [periodic] * d if np.ndim(periodic) == 0 else tuple(periodic)
    # --- Pad to 3D ---
    np_positions_3d = np.pad(np_positions, ((0, 0), (0, 3 - d)))
    if domain is None:  # we cannot pass None to matscipy if d < 3, else cells get zero volume
        domain = b.min(positions, 0), b.max(positions, 0)
    domain_origin_3d = np.pad(b.numpy(domain[0]), (0, 3 - d), constant_values=-max_cutoff / 2)
    domain_size_3d = np.diag(b.numpy(b.maximum(np.pad(b.numpy(domain[1] - domain[0]), (0, 3 - d)), max_cutoff)))
    # --- Run neighbor search ---
    row, col = neighbour_list('ij', positions=np_positions_3d, cutoff=cutoff, cell_origin=domain_origin_3d, cell=domain_size_3d, pbc=periodic)
    row = row.astype(to_numpy_dtype(index_dtype), copy=False)
    col = col.astype(to_numpy_dtype(index_dtype), copy=False)
    pos_self = b.gather(positions, row, 0)
    pos_neighbors = b.gather(positions, col, 0)
    deltas = pos_neighbors - pos_self
    # --- Add self as neighbor with distance 0 ---
    self_indices = np.arange(pos_count, dtype=to_numpy_dtype(index_dtype))
    self_deltas = b.zeros((pos_count, d), b.dtype(positions))
    row = b.concat([self_indices, row], 0)
    col = b.concat([self_indices, col], 0)
    deltas = b.concat([self_deltas, deltas], 0)
    return row, col, deltas

# def find_neighbors_jax_md(positions: TensorType,
#                           cutoff: Union[float, TensorType],
#                           domain: Optional[Tuple[TensorType, TensorType]] = None,
#                           periodic: Union[bool, Tuple[bool, ...]] = False,
#                           index_dtype=DType(int, 32)):
#     from jax_md import space, partition
#     displacement_fn, shift_fn = space.periodic(1.0)
#     jaxmd_nl_fn = partition.neighbor_list(
#         displacement_fn,
#         box=box,
#         r_cutoff=cutoffs[i],
#         format=partition.NeighborListFormat.Sparse,
#         capacity_multiplier=1.0
#     )
#     jaxmd_nl = jaxmd_nl_fn.allocate(poss[i])
#     jaxmd_nl = jaxmd_nl.update(poss[i])
