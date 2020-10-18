import numpy as np

from .backend import DYNAMIC_BACKEND as math
from . import _extrapolation as extrapolation
from ._tensors import Tensor, NativeTensor, combined_shape, TensorStack


class SparseLinearOperation(Tensor):

    def __init__(self, source: Tensor, dependency_matrix, shape):
        self.source = source
        self.dependency_matrix = dependency_matrix
        self._shape = shape

    def native(self, order=None):
        native_source = math.reshape(self.source.native(), (self.source.shape.batch.volume, self.source.shape.non_batch.volume))
        native = math.matmul(self.dependency_matrix, native_source)
        new_shape = self.source.shape.batch.combined(self._shape)
        native = math.reshape(native, new_shape.sizes)
        return NativeTensor(native, new_shape).native(order)

    @property
    def dtype(self):
        return self.source.dtype

    @property
    def shape(self):
        return self._shape

    def _with_shape_replaced(self, new_shape):
        raise NotImplementedError()

    def _getitem(self, selection_dict):
        indices = NativeTensor(math.reshape(math.range(self.shape.volume), self.shape.sizes), self.shape)
        selected_indices = indices[selection_dict]
        selected_indices_native = math.flatten(selected_indices.native())
        selected_deps = math.gather(self.dependency_matrix, (selected_indices_native, slice(None)))
        return SparseLinearOperation(self.source, selected_deps, selected_indices.shape)

    def unstack(self, dimension):
        raise NotImplementedError()

    def _op1(self, native_function):
        deps = native_function(self.dependency_matrix)
        return SparseLinearOperation(self.source, deps, self._shape)

    def _op2(self, other, native_function):
        if isinstance(other, SparseLinearOperation):
            assert self.source is other.source
            assert self._shape == other._shape
            new_matrix = native_function(self.dependency_matrix, other.dependency_matrix)
            return SparseLinearOperation(self.source, new_matrix, self._shape)
        else:
            other = self._tensor(other)
            broadcast_shape = combined_shape(self, other)
            if other.shape.volume > 1:
                additional_dims = broadcast_shape.without(self._shape)
                if len(additional_dims) == 0:
                    flat = math.flatten(other.native(broadcast_shape.names))
                    vertical = math.expand_dims(flat, -1)
                    new_matrix = native_function(self.dependency_matrix, vertical)  # this can cause matrix to become dense...
                elif len(additional_dims) == 1:
                    others = other.unstack(additional_dims.names[0])
                    results = [self._op2(o, native_function) for o in others]
                    return TensorStack(results, additional_dims.names[0], additional_dims.types[0])
                else:
                    raise NotImplementedError()
            else:
                scalar = other[{dim: 0 for dim in other.shape.names}].native()
                new_matrix = native_function(self.dependency_matrix, scalar)
            return SparseLinearOperation(self.source, new_matrix, broadcast_shape)


def as_sparse_linear_operation(tensor: Tensor):
    tracking_shape = tensor.shape.non_batch
    idx = math.range(tracking_shape.volume)
    ones = math.ones_like(idx)
    sparse_diag = math.choose_backend(tensor.native()).sparse_tensor([idx, idx], ones, shape=(tracking_shape.volume,) * 2)
    return SparseLinearOperation(tensor, sparse_diag, tracking_shape)


def sum_operators(operators):
    for o in operators[1:]:
        assert isinstance(o, SparseLinearOperation)
        assert o.source is operators[0].source
    new_matrix = math.sum([o.dependency_matrix for o in operators], axis=0)
    return SparseLinearOperation(operators[0].source, new_matrix, operators[0].shape)
