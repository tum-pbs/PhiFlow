import warnings
from numbers import Number
from typing import List, Callable, Tuple

import numpy as np
import scipy.sparse

from ._shape import Shape, non_batch, merge_shapes, instance, batch, non_instance, shape, channel, spatial, DimFilter, concat_shapes, EMPTY_SHAPE, dual, DUAL_DIM, SPATIAL_DIM
from ._magic_ops import concat, pack_dims, expand, rename_dims
from ._tensors import Tensor, TensorStack, CollapsedTensor, NativeTensor, cached, wrap
from .backend import choose_backend, NUMPY
from .backend._dtype import DType


class SparseCoordinateTensor(Tensor):

    def __init__(self, indices: Tensor, values: Tensor, dense_shape: Shape, can_contain_double_entries: bool, indices_sorted: bool):
        """
        Construct a sparse tensor with any number of sparse, dense and batch dimensions.

        Args:
            indices: `Tensor` encoding the positions of stored values. It has the following dimensions:

                * One instance dimension exactly matching the instance dimension on `values`.
                  It enumerates the positions of stored entries.
                * One channel dimension called `vector`.
                  Its item names must match the dimension names of `dense_shape` but the order can be arbitrary.
                * Any number of batch dimensions

            values: `Tensor` containing the stored values at positions given by `indices`. It has the following dimensions:

                * One instance dimension exactly matching the instance dimension on `indices`.
                  It enumerates the values of stored entries.
                * Any number of channel dimensions if multiple values are stored at each index.
                * Any number of batch dimensions

            dense_shape: Dimensions listed in `indices`.
                The order can differ from the item names of `indices`.
            can_contain_double_entries: Whether some indices might occur more than once.
                If so, values at the same index will be summed.
            indices_sorted: Whether the indices are sorted in ascending order given the dimension order of the item names of `indices`.
        """
        assert instance(indices), "indices must have an instance dimension"
        assert 'vector' in indices.shape, "indices must have a vector dimension"
        assert set(indices.vector.item_names) == set(dense_shape.names), "The 'vector' dimension of indices must list the dense dimensions as item names"
        assert indices.dtype.kind == int, f"indices must have dtype=int but got {indices.dtype}"
        self._shape = merge_shapes(dense_shape, batch(indices), non_instance(values))
        self._dense_shape = dense_shape
        self._indices = indices
        self._values = values
        self._can_contain_double_entries = can_contain_double_entries
        self._indices_sorted = indices_sorted

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def dtype(self) -> DType:
        return self._values.dtype

    def native(self, order: str or tuple or list or Shape = None):
        raise RuntimeError("Sparse tensors do not have a native representation. Use math.dense(tensor).native() instead")

    @property
    def _is_tracer(self) -> bool:
        return self._indices._is_tracer or self._values._is_tracer

    def _natives(self) -> tuple:
        indices_const = self._indices.default_backend is not self._values.default_backend
        if indices_const:
            return self._values._natives()  # If we return NumPy arrays, they might get converted in function transformations
        else:
            return self._values._natives() + self._indices._natives()

    def _spec_dict(self) -> dict:
        indices_const = self._indices.default_backend is not self._values.default_backend
        return {'type': SparseCoordinateTensor,
                'shape': self._shape,
                'dense_shape': self._dense_shape,
                'indices': self._indices if indices_const else self._indices._spec_dict(),
                'values': self._values._spec_dict(),
                'can_contain_double_entries': self._can_contain_double_entries,
                'indices_sorted': self._indices_sorted}

    @classmethod
    def _from_spec_and_natives(cls, spec: dict, natives: list):
        values = spec['values']['type']._from_spec_and_natives(spec['values'], natives)
        indices_or_spec = spec['indices']
        if isinstance(indices_or_spec, Tensor):
            indices = indices_or_spec
        else:
            indices = spec['indices']['type']._from_spec_and_natives(spec['indices'], natives)
        return SparseCoordinateTensor(indices, values, spec['dense_shape'], spec['can_contain_double_entries'], spec['indices_sorted'])

    def _native_coo_components(self, col_dims: DimFilter, matrix=False):
        col_dims = self._shape.only(col_dims)
        row_dims = self._dense_shape.without(col_dims)
        row_idx_packed, col_idx_packed = self._pack_indices(row_dims, col_dims)
        from phi.math import reshaped_native
        ind_batch = batch(self._indices)
        channels = non_instance(self._values).without(ind_batch)
        if matrix:
            native_indices = self.default_backend.stack([row_idx_packed, col_idx_packed], -1)
            native_shape = (row_dims.volume, col_dims.volume)
        else:
            native_indices = reshaped_native(self._indices, [ind_batch, instance, 'vector'], force_expand=True)
            native_shape = self._dense_shape.sizes
        native_values = reshaped_native(self._values, [ind_batch, instance, channels])
        return ind_batch, channels, native_indices, native_values, native_shape

    def _pack_indices(self, row_dims: Shape, col_dims: Shape):
        assert self._indices.default_backend is NUMPY, "Can only compress NumPy indices as of yet"
        assert row_dims in self._dense_shape, f"Can only compress sparse dims but got {row_dims} which contains non-sparse dims"
        from ._ops import reshaped_native
        row_idx = self._indices[row_dims.names]
        col_idx = self._indices[self._dense_shape.without(row_dims).names]
        # ToDo if not row_dims: idx = [0]
        row_idx_packed = np.ravel_multi_index(reshaped_native(row_idx, [channel, batch, instance]), row_dims.sizes)
        col_idx_packed = np.ravel_multi_index(reshaped_native(col_idx, [channel, batch, instance]), col_dims.sizes)
        return row_idx_packed, col_idx_packed

    def compress_rows(self):
        return self.compress(self._dense_shape.non_dual)

    def compress_cols(self):
        return self.compress(self._dense_shape.dual)

    def compress(self, dims: DimFilter):
        c_dims = self._shape.only(dims, reorder=True)
        u_dims = self._dense_shape.without(c_dims)
        c_idx_packed, u_idx_packed = self._pack_indices(c_dims, u_dims)
        # --- Use scipy.sparse.csr_matrix to reorder values ---
        idx = np.arange(1, c_idx_packed.shape[-1] + 1)  # start indexing at 1 since 0 might get removed
        scipy_csr = scipy.sparse.csr_matrix((idx, (c_idx_packed[0], u_idx_packed[0])), shape=(c_dims.volume, u_dims.volume))
        assert c_idx_packed.shape[1] == len(scipy_csr.data), "Failed to create CSR matrix because the CSR matrix contains fewer non-zero values than COO. This can happen when the `x` tensor is too small for the stencil."
        # --- Construct CompressedSparseMatrix ---
        entries_dim = instance(self._values).name
        values = self._values[{entries_dim: wrap(scipy_csr.data - 1, instance(entries_dim))}]  # Change order accordingly
        indices = wrap(scipy_csr.indices, instance(entries_dim))
        pointers = wrap(scipy_csr.indptr, instance('pointers'))
        return CompressedSparseMatrix(indices, pointers, values, u_dims, c_dims)

    def __pack_dims__(self, dims: Tuple[str, ...], packed_dim: Shape, pos: int or None, **kwargs) -> 'Tensor':
        dims = self._shape.only(dims)
        assert self._dense_shape in dims, "Can only pack sparse dimensions on SparseCoordinateTensor"
        assert self._indices.default_backend is NUMPY, "Can only pack NumPy indices as of yet"
        from ._ops import reshaped_native
        idx = self._indices.vector[dims.names]
        idx_packed = np.ravel_multi_index(reshaped_native(idx, [channel, instance]), dims.sizes)
        idx_packed = expand(wrap(idx_packed, instance(self._indices)), channel(vector=packed_dim.name))
        indices = concat([self._indices.vector[self._dense_shape.without(dims).names], idx_packed], 'vector')
        dense_shape = concat_shapes(self._dense_shape.without(dims), packed_dim.with_size(dims.volume))
        idx_sorted = self._indices_sorted and False  # ToDo still sorted if dims are ordered correctly and no other dim in between and inserted at right point
        return SparseCoordinateTensor(indices, self._values, dense_shape, self._can_contain_double_entries, idx_sorted)

    def _with_shape_replaced(self, new_shape: Shape):
        assert self._shape.rank == new_shape.rank
        dense_shape = new_shape[self._shape.indices(self._dense_shape)]
        new_item_names = new_shape[self._shape.indices(self._indices.shape.get_item_names('vector'))].names
        indices = self._indices._with_shape_replaced(self._indices.shape.replace(self._shape, new_shape).with_dim_size('vector', new_item_names))
        values = self._values._with_shape_replaced(self._values.shape.replace(self._shape, new_shape))
        return SparseCoordinateTensor(indices, values, dense_shape, self._can_contain_double_entries, self._indices_sorted)


class CompressedSparseMatrix(Tensor):

    def __init__(self, indices: Tensor, pointers: Tensor, values: Tensor, uncompressed_dims: Shape, compressed_dims: Shape, uncompressed_offset: int = None):
        """

        Args:
            indices: indices must be sorted in ascending order by compressed_dim and other sparse dims.
                Must have one or multiple instance dimensions and can have any number of batch dimensions.
                No spatial and channel dimensions allowed.
            pointers:
            values:
            compressed_dims: Sparse dimensions with compressed pointer representation.
                Only one pointer array is used per matrix, i.e. the dimensions are packed internally.
                These dimensions are indexed by `pointers`.
            uncompressed_dims: Sparse dimensions with full index storage.
                These dimensions are indexed by `indices`.
            uncompressed_offset: For sliced sparse tensors.
                If `None`, indicates that all entries lie within bounds.
                If an `int`, indicate that this is a slice of a larger compressed sparse matrix.
                Indices actually refer to `indices - uncompressed_offset` within this matrix, i.e. they may reference phantom values to the left or right of the matrix.
                The `values` corresponding to phantom entries must all be 0.
                The size of the slice is given by `compressed_dims.volume`.
        """
        assert instance(indices), "indices must have an instance dimension"
        assert instance(pointers), "pointers must have an instance dimension"
        assert instance(values) == instance(indices), "Instance dimensions of values and indices must match exactly"
        assert not channel(indices) and not spatial(indices), f"channel and spatial dimensions not allowed on indices but got {shape(indices)}"
        assert not channel(pointers) and not spatial(pointers), f"channel and spatial dimensions not allowed on pointers but got {shape(pointers)}"
        assert uncompressed_dims.isdisjoint(compressed_dims), f"Dimensions cannot be compressed and uncompressed at the same time but got compressed={compressed_dims}, uncompressed={uncompressed_dims}"
        assert instance(pointers).size == compressed_dims.volume + 1
        self._shape = merge_shapes(compressed_dims, uncompressed_dims, batch(indices), batch(pointers), non_instance(values))
        self._indices = indices
        self._pointers = pointers
        self._values = values
        self._uncompressed_dims = uncompressed_dims
        self._compressed_dims = compressed_dims
        self._uncompressed_offset = uncompressed_offset

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def sparse_dims(self):
        return self._compressed_dims & self._uncompressed_dims

    @property
    def sparsity_batch(self):
        return batch(self._indices) & batch(self._pointers)

    @property
    def dtype(self) -> DType:
        return self._values.dtype

    @property
    def _is_tracer(self) -> bool:
        return self._values._is_tracer or self._indices._is_tracer or self._pointers._is_tracer

    def _natives(self) -> tuple:
        indices_const = self._indices.default_backend is not self._values.default_backend
        pointers_const = self._pointers.default_backend is not self._values.default_backend
        result = self._values._natives()
        if not indices_const:
            result += self._indices._natives()
        if not pointers_const:
            result += self._pointers._natives()
        return result

    def _spec_dict(self) -> dict:
        indices_const = self._indices.default_backend is not self._values.default_backend
        pointers_const = self._pointers.default_backend is not self._values.default_backend
        return {'type': CompressedSparseMatrix,
                'shape': self._shape,
                'values': self._values._spec_dict(),
                'indices': self._indices if indices_const else self._indices._spec_dict(),
                'pointers': self._pointers if pointers_const else self._pointers._spec_dict(),
                'uncompressed_dims': self._uncompressed_dims,
                'compressed_dims': self._compressed_dims,
                'uncompressed_offset': self._uncompressed_offset}

    @classmethod
    def _from_spec_and_natives(cls, spec: dict, natives: list):
        values = spec['values']['type']._from_spec_and_natives(spec['values'], natives)
        indices_or_spec = spec['indices']
        if isinstance(indices_or_spec, Tensor):
            indices = indices_or_spec
        else:
            indices = spec['indices']['type']._from_spec_and_natives(spec['indices'], natives)
        pointers_or_spec = spec['pointers']
        if isinstance(pointers_or_spec, Tensor):
            pointers = pointers_or_spec
        else:
            pointers = spec['pointers']['type']._from_spec_and_natives(spec['pointers'], natives)
        return CompressedSparseMatrix(indices, pointers, values, spec['uncompressed_dims'], spec['compressed_dims'], spec['uncompressed_offset'])

    def _getitem(self, selection: dict) -> 'Tensor':
        batch_selection = {dim: selection[dim] for dim in self._shape.only(tuple(selection)).names}
        indices = self._indices[batch_selection]
        pointers = self._pointers[batch_selection]
        values = self._values[batch_selection]
        uncompressed = self._uncompressed_dims
        compressed = self._compressed_dims
        uncompressed_offset = self._uncompressed_offset
        if compressed.only(tuple(selection)):
            if compressed.rank > 1:
                raise NotImplementedError
            ptr_sel = selection[compressed.name]
            if isinstance(ptr_sel, int):
                raise NotImplementedError(f"Slicing with int not yet supported for sparse tensors. Use a range instead, e.g. [{ptr_sel}:{ptr_sel+1}] instead of [{ptr_sel}]")
            elif isinstance(ptr_sel, slice):
                assert ptr_sel.step in (None, 1), f"Only step size 1 supported for sparse indexing but got {ptr_sel.step}"
                if batch(indices):
                    raise NotImplementedError("Slicing not yet supported for batched sparse tensors")
                start = ptr_sel.start or 0
                stop = uncompressed.volume if ptr_sel.stop is None else ptr_sel.stop
                pointers = pointers[start:stop+1]
                indices = indices[{instance(indices).name: slice(int(pointers[0]), int(pointers[-1]))}]
                values = values[{instance(values).name: slice(int(pointers[0]), int(pointers[-1]))}]
                pointers -= pointers[0]
                compressed = compressed.after_gather({compressed.name: ptr_sel})
            else:
                raise NotImplementedError
        if uncompressed.only(tuple(selection)):
            if self._uncompressed_dims.rank > 1:
                raise NotImplementedError
            ind_sel = selection[uncompressed.name]
            if isinstance(ind_sel, int):
                raise NotImplementedError(f"Slicing with int not yet supported for sparse tensors. Use a range instead, e.g. [{ind_sel}:{ind_sel+1}] instead of [{ind_sel}]")
            elif isinstance(ind_sel, slice):
                assert ind_sel.step in (None, 1), f"Only step size 1 supported for sparse indexing but got {ind_sel.step}"
                start = ind_sel.start or 0
                stop = uncompressed.volume if ind_sel.stop is None else ind_sel.stop
                keep = (start <= self._indices) & (self._indices < stop)
                from phi.math import where
                values = where(keep, values, 0)
                uncompressed_offset = start
                uncompressed = uncompressed.after_gather({uncompressed.name: ind_sel})
            else:
                raise NotImplementedError
        return CompressedSparseMatrix(indices, pointers, values, uncompressed, compressed, uncompressed_offset)

    def __concat__(self, tensors: tuple, dim: str, **kwargs) -> 'CompressedSparseMatrix':
        if not all(isinstance(t, CompressedSparseMatrix) for t in tensors):
            return NotImplemented
        if dim == self._compressed_dims[0].name:
            indices = concat([t._indices for t in tensors], instance(self._indices), **kwargs)
            values = concat([t._values for t in tensors], instance(self._values), **kwargs)
            pointers = []
            pointer_offset = 0
            for i, t in enumerate(tensors):
                pointers.append((t._pointers[1:] if i else t._pointers) + pointer_offset)
                pointer_offset += t._pointers[-1]
            assert pointer_offset == instance(indices).volume
            pointers = concat(pointers, instance(self._pointers))
            compressed = self._compressed_dims.with_dim_size(dim, sum(t.shape.get_size(dim) for t in tensors))
            return CompressedSparseMatrix(indices, pointers, values, self._uncompressed_dims, compressed, self._uncompressed_offset)
        elif dim == self._uncompressed_dims[0].name:
            if all(t._indices is self._indices and t._pointers is self._pointers for t in tensors):
                # ToDo test if offsets match and ordered correctly
                from ._ops import sum_
                values = sum_([t._values for t in tensors], '0')
                uncompressed = self._uncompressed_dims.with_dim_size(dim, sum(t.shape.get_size(dim) for t in tensors))
                return CompressedSparseMatrix(self._indices, self._pointers, values, uncompressed, self._compressed_dims, uncompressed_offset=None)
            else:
                raise NotImplementedError("concatenating arbitrary compressed sparse tensors along uncompressed dim is not yet supported")
        else:
            raise NotImplementedError("concatenating compressed sparse tensors along non-sparse dims not yet supported")

    def _op1(self, native_function):
        return self._with_values(self._values._op1(native_function))

    def _op2(self, other, operator: Callable, native_function: Callable, op_name: str = 'unknown', op_symbol: str = '?') -> 'Tensor':
        other_shape = shape(other)
        affects_only_values = self.sparse_dims not in other_shape and non_instance(self._indices).isdisjoint(other_shape)
        if affects_only_values:
            return self._with_values(operator(self._values, other))
        elif isinstance(other, CompressedSparseMatrix):
            if other._indices is self._indices and other._pointers is self._pointers:
                return self._with_values(operator(self._values, other._values))
            elif op_symbol == '+':
                raise NotImplementedError("Compressed addition not yet implemented")
            else:
                # convert to COO, then perform operation
                raise NotImplementedError
        raise NotImplementedError

    def _with_values(self, new_values: Tensor):
        return CompressedSparseMatrix(self._indices, self._pointers, new_values, self._uncompressed_dims, self._compressed_dims)

    def _with_shape_replaced(self, new_shape: Shape):
        assert self._shape.rank == new_shape.rank
        raise NotImplementedError

    def _native_csr_components(self):
        from phi.math import reshaped_native
        ind_batch = batch(self._indices) & batch(self._pointers)
        channels = non_instance(self._values).without(ind_batch)
        native_indices = reshaped_native(self._indices, [ind_batch, instance], force_expand=True)
        native_pointers = reshaped_native(self._pointers, [ind_batch, instance], force_expand=True)
        native_values = reshaped_native(self._values, [ind_batch, instance, channels])
        native_shape = self._compressed_dims.volume, self._uncompressed_dims.volume
        if self._uncompressed_offset is not None:
            native_indices -= self._uncompressed_offset
            native_indices = choose_backend(native_indices).clip(native_indices, 0, self._uncompressed_dims.volume - 1)
        return ind_batch, channels, native_indices, native_pointers, native_values, native_shape

    def native(self, order: str or tuple or list or Shape = None):
        raise RuntimeError("Sparse tensors do not have a native representation. Use math.dense(tensor).native() instead")

    def __pack_dims__(self, dims: Tuple[str, ...], packed_dim: Shape, pos: int or None, **kwargs) -> 'Tensor':
        assert all(d in self._shape for d in dims)
        dims = self._shape.only(dims, reorder=True)
        if dims.only(self._compressed_dims).is_empty:  # pack cols
            assert self._uncompressed_dims.are_adjacent(dims), f"Can only compress adjacent dims but got {dims} for matrix {self._shape}"
            uncompressed_dims = self._uncompressed_dims.replace(dims, packed_dim.with_size(dims.volume))
            return CompressedSparseMatrix(self._indices, self._pointers, self._values, uncompressed_dims, self._compressed_dims, self._uncompressed_offset)
        elif dims.only(self._uncompressed_dims).is_empty:   # pack rows
            assert self._compressed_dims.are_adjacent(dims), f"Can only compress adjacent dims but got {dims} for matrix {self._shape}"
            compressed_dims = self._compressed_dims.replace(dims, packed_dim.with_size(dims.volume))
            return CompressedSparseMatrix(self._indices, self._pointers, self._values, self._uncompressed_dims, compressed_dims, self._uncompressed_offset)
        else:
            raise NotImplementedError(f"Cannot pack dimensions from both columns and rows with compressed sparse matrices but got {dims}")

def sparse_dims(x: Tensor) -> Shape:
    """
    Returns the dimensions of a `Tensor` that are explicitly stored in a sparse format.

    Args:
        x: Any `Tensor`

    Returns:
        `Shape`
    """
    if isinstance(x, SparseCoordinateTensor):
        return x._dense_shape
    elif isinstance(x, CompressedSparseMatrix):
        return concat_shapes(x._compressed_dims, x._uncompressed_dims)
    else:
        return EMPTY_SHAPE


def get_sparsity(x: Tensor):
    """
    Fraction of values currently stored on disk for the given `Tensor` `x`.
    For sparse tensors, this is `nnz / shape`.

    This is a lower limit on the number of values that will need to be processed for operations involving `x`.
    The actual number is often higher since many operations require data be laid out in a certain format.
    In these cases, missing values, such as zeros, are filled in before the operation.

    The following operations may return tensors whose values are only partially stored:

    * `phi.math.expand()`
    * `phi.math.pairwise_distance()` with `max_distance` set.
    * Tracers used in `phi.math.jit_compile_linear()`
    * Stacking any of the above.

    Args:
        x: `Tensor`

    Returns:
        The number of values that are actually stored on disk.
        This does not include additional information, such as position information / indices.
        For sparse matrices, this is equal to the number of nonzero values.
    """
    # ToDo this does not give the correct result for linear tracers since the matrix shape is not taken into account
    return sum([t.shape.volume for t in stored_values(x)]) / x.shape.volume


def stored_values(x: Tensor) -> List[Tensor]:
    """
    Returns the values currently stored on disk for the given `Tensor` `x`.

    Some operations may require non-stored values to be explicitly stored, or they may be filled in for performance reasons.

    Args:
        x: `Tensor`

    Returns:
        List of `Tensor`s representing all values stored to represent `x`.
    """
    if isinstance(x, NativeTensor):
        return [x]
    elif isinstance(x, CollapsedTensor):
        return [cached(x)] if x.is_cached else stored_values(x._inner)
    elif isinstance(x, TensorStack):
        return [cached(x)] if x.is_cached else sum([stored_values(t) for t in x._tensors], [])
    elif isinstance(x, CompressedSparseMatrix):
        return [x._values]
    elif isinstance(x, SparseCoordinateTensor):
        if x._can_contain_double_entries:
            warnings.warn(f"Sparsity of sparse tensor {x.shape} is unknown as multiple values can reference the same position.")
        return [x._values]
    else:
        from phi.math._functional import ShiftLinTracer
        if isinstance(x, ShiftLinTracer):
            return sum([stored_values(v) for v in x.val.values()], [])
        raise ValueError(x)


def dense(x: Tensor) -> Tensor:
    """
    Convert a sparse tensor representation to an equivalent dense one in which all values are explicitly stored contiguously in memory.

    Args:
        x: Any `Tensor`.
            Python primitives like `float`, `int` or `bool` will be converted to `Tensors` in the process.

    Returns:
        Dense tensor.
    """
    from phi.math import reshaped_tensor
    if isinstance(x, SparseCoordinateTensor):
        from ._ops import scatter, zeros
        base_grid = zeros(x.shape._with_types(SPATIAL_DIM), dtype=x.dtype)
        result_sp = scatter(base_grid, x._indices, x._values, mode='add', outside_handling='undefined')
        result = rename_dims(result_sp, shape, x.shape)
        return result
    elif isinstance(x, CompressedSparseMatrix):
        ind_batch, channels, native_indices, native_pointers, native_values, native_shape = x._native_csr_components()
        native_dense = x.default_backend.csr_to_dense(native_indices, native_pointers, native_values, native_shape)
        return reshaped_tensor(native_dense, [ind_batch, x._compressed_dims, x._uncompressed_dims, channels])
    elif isinstance(x, NativeTensor):
        return x
    elif isinstance(x, Tensor):
        return cached(x)
    elif isinstance(x, (Number, bool)):
        return wrap(x)


def dot_compressed_dense(compressed: CompressedSparseMatrix, cdims: Shape, dense: Tensor, ddims: Shape):
    from phi.math import reshaped_native, reshaped_tensor
    backend = choose_backend(*compressed._natives() + dense._natives())
    if compressed._uncompressed_dims in cdims:  # proper matrix-vector multiplication
        ind_batch, channels, native_indices, native_pointers, native_values, native_shape = compressed._native_csr_components()
        rhs_channels = shape(dense).without(ddims).without(channels)
        dense_native = reshaped_native(dense, [ind_batch, ddims, channels, rhs_channels], force_expand=True)
        result_native = backend.mul_csr_dense(native_indices, native_pointers, native_values, native_shape, dense_native)
        result = reshaped_tensor(result_native, [ind_batch, channels, compressed._compressed_dims, rhs_channels])
        return result
    else:  # transposed matrix vector multiplication. This is inefficient
        raise NotImplementedError("Transposed sparse matrix multiplication not yet implemented")


def dot_coordinate_dense(sparse: SparseCoordinateTensor, sdims: Shape, dense: Tensor, ddims: Shape):
    from phi.math import reshaped_native, reshaped_tensor
    backend = choose_backend(*sparse._natives() + dense._natives())
    ind_batch, channels, native_indices, native_values, native_shape = sparse._native_coo_components(sdims, matrix=True)
    rhs_channels = shape(dense).without(ddims).without(channels)
    dense_native = reshaped_native(dense, [ind_batch, ddims, channels, rhs_channels], force_expand=True)
    result_native = backend.mul_coo_dense(native_indices, native_values, native_shape, dense_native)
    result = reshaped_tensor(result_native, [ind_batch, channels, sparse._dense_shape.without(sdims), rhs_channels])
    return result


def native_matrix(value: Tensor):
    cols = dual(value)
    rows = non_batch(value).non_dual
    if isinstance(value, SparseCoordinateTensor):
        ind_batch, channels, indices, values, shape = value._native_coo_components(dual, matrix=True)
        if ind_batch.volume > 1 or channels.volume > 1:
            return value.default_backend.sparse_coo_tensor_batched(indices, values, shape)
        else:
            return value.default_backend.sparse_coo_tensor(indices[0], values[0, :, 0], shape)
    elif isinstance(value, CompressedSparseMatrix):
        assert not non_instance(value._values), f"native_matrix does not support vector-valued matrices. Vector dims: {non_batch(value).without(sparse_dims(value))}"
        ind_batch, channels, indices, pointers, values, shape = value._native_csr_components()
        if dual(value._uncompressed_dims):  # CSR
            assert not dual(value._compressed_dims), "Dual dimensions on both compressed and uncompressed dimensions"
            if ind_batch.volume > 1 or channels.volume > 1:
                return value.default_backend.csr_matrix_batched(indices, pointers, values, shape)
            else:
                return value.default_backend.csr_matrix(indices[0], pointers[0], values[0, :, 0], shape)
        else:  # CSC
            assert not dual(value._uncompressed_dims)
            if ind_batch.volume > 1 or channels.volume > 1:
                return value.default_backend.csc_matrix_batched(pointers, indices, values, shape)
            else:
                return value.default_backend.csc_matrix(pointers[0], indices[0], values[0, :, 0], shape)
    else:
        v = pack_dims(value, rows, channel('_row'))
        v = pack_dims(v, cols, channel('_col'))
        from ._ops import reshaped_native
        return reshaped_native(v, [batch, '_row', '_col'])