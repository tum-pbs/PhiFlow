import warnings
from numbers import Number
from typing import List, Callable

from ._shape import Shape, non_batch, merge_shapes, instance, batch, non_instance, shape, channel, spatial
from ._magic_ops import concat
from ._tensors import Tensor, TensorStack, CollapsedTensor, NativeTensor, cached, wrap
from .backend import choose_backend, Backend
from .backend._dtype import DType


class SparseCoordinateTensor(Tensor):

    def __init__(self, indices: Tensor, values: Tensor, dense_shape: Shape, can_contain_double_entries: bool, indices_sorted: bool):
        assert instance(indices), "indices must have an instance dimension"
        assert 'vector' in indices.shape, "indices must have a vector dimension"
        assert indices.vector.item_names is not None and len(indices.vector.item_names) == non_batch(values).non_channel.rank, "The 'vector' dimension of indices must list the dense dimensions as item names"
        self._shape = merge_shapes(dense_shape, batch(indices), non_instance(values))
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


class CompressedSparseTensor(Tensor):

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
        assert uncompressed_dims.only(compressed_dims).is_empty, f"Dimensions cannot be compressed and uncompressed at the same time but got compressed={compressed_dims}, uncompressed={uncompressed_dims}"
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
        return self._values._natives() + self._indices._natives() + self._pointers._natives()

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
        return CompressedSparseTensor(indices, pointers, values, uncompressed, compressed, uncompressed_offset)

    def __concat__(self, tensors: tuple, dim: str, **kwargs) -> 'CompressedSparseTensor':
        if not all(isinstance(t, CompressedSparseTensor) for t in tensors):
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
            return CompressedSparseTensor(indices, pointers, values, self._uncompressed_dims, compressed, self._uncompressed_offset)
        elif dim == self._uncompressed_dims[0].name:
            if all(t._indices is self._indices and t._pointers is self._pointers for t in tensors):
                # ToDo test if offsets match and ordered correctly
                from ._ops import sum_
                values = sum_([t._values for t in tensors], '0')
                uncompressed = self._uncompressed_dims.with_dim_size(dim, sum(t.shape.get_size(dim) for t in tensors))
                return CompressedSparseTensor(self._indices, self._pointers, values, uncompressed, self._compressed_dims, uncompressed_offset=None)
            else:
                raise NotImplementedError("concatenating arbitrary compressed sparse tensors along uncompressed dim is not yet supported")
        else:
            raise NotImplementedError("concatenating compressed sparse tensors along non-sparse dims not yet supported")

    def _op1(self, native_function):
        return self._with_values(self._values._op1(native_function))

    def _op2(self, other, operator: Callable, native_function: Callable, op_name: str = 'unknown', op_symbol: str = '?') -> 'Tensor':
        other_shape = shape(other)
        affects_only_values = self.sparse_dims not in other_shape and non_instance(self._indices).only(other_shape).is_empty
        if affects_only_values:
            return self._with_values(operator(self._values, other))
        elif isinstance(other, CompressedSparseTensor):
            if other._indices is self._indices and other._pointers is self._pointers:
                return self._with_values(operator(self._values, other._values))
            elif op_symbol == '+':
                raise NotImplementedError("Compressed addition not yet implemented")
            else:
                # convert to COO, then perform operation
                raise NotImplementedError
        raise NotImplementedError

    def _with_values(self, new_values: Tensor):
        return CompressedSparseTensor(self._indices, self._pointers, new_values, self._uncompressed_dims, self._compressed_dims)

    def _native_csr_components(self):
        from phi.math import reshaped_native
        ind_batch = batch(self._indices & self._pointers)
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
    elif isinstance(x, CompressedSparseTensor):
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
        raise NotImplementedError
        native_dense = x.default_backend.coo_to_dense()
    elif isinstance(x, CompressedSparseTensor):
        ind_batch, channels, native_indices, native_pointers, native_values, native_shape = x._native_csr_components()
        native_dense = x.default_backend.csr_to_dense(native_indices, native_pointers, native_values, native_shape)
        return reshaped_tensor(native_dense, [ind_batch, x._compressed_dims, x._uncompressed_dims, channels])
    elif isinstance(x, NativeTensor):
        return x
    elif isinstance(x, Tensor):
        return cached(x)
    elif isinstance(x, (Number, bool)):
        return wrap(x)


def dot_compressed_dense(compressed: CompressedSparseTensor, cdims: Shape, dense: Tensor, ddims: Shape):
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
