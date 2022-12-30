import warnings
from numbers import Number
from typing import List, Callable

from ._shape import Shape, non_batch, merge_shapes, instance, batch, non_instance, shape, channel, spatial
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

    def __init__(self, indices: Tensor, pointers: Tensor, values: Tensor, uncompressed_dims: Shape, compressed_dims: Shape):
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
        if self._compressed_dims.only(tuple(selection)):
            raise NotImplementedError
        if self._uncompressed_dims.only(tuple(selection)):
            raise NotImplementedError
        batch_selection = {dim: selection[dim] for dim in self._shape.only(tuple(selection)).names}
        return CompressedSparseTensor(self._indices[batch_selection], self._pointers[batch_selection], self._values[batch_selection], self._uncompressed_dims, self._compressed_dims)

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
        native_shape = self._uncompressed_dims.volume, self._compressed_dims.volume
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
        dense_native = reshaped_native(dense, [ind_batch, channels, ddims, rhs_channels], force_expand=True)
        if backend.supports(Backend.mul_csr_dense):
            result_native = backend.mul_csr_dense(native_indices, native_pointers, native_values, native_shape, dense_native)
        else:
            native_coo_indices = backend.csr_to_coo(native_indices, native_pointers)
            result_native = backend.mul_coo_dense(native_coo_indices, native_values, native_shape, dense_native)
        result = reshaped_tensor(result_native, [ind_batch, channels, compressed._compressed_dims, rhs_channels])
        return result
    else:  # transposed matrix vector multiplication. This is inefficient
        raise NotImplementedError("Transposed sparse matrix multiplication not yet implemented")
