from functools import reduce

import numpy as np

from .backend import math as native_math, choose_backend
from ._shape import EMPTY_SHAPE, Shape, shape_from_dict
from ._tensors import Tensor, NativeTensor, combined_shape, TensorStack
from . import _functions as math


def simplify_add(val: dict):
    result = {}
    for shift, values in val.items():
        shift = shift.non_zero
        if shift in result:
            result[shift] += values
        else:
            result[shift] = values
    return result


class ShiftLinOp(Tensor):

    def __init__(self, source: Tensor, values_by_shift: dict, shape: Shape):
        """

        :param source: placeholder tensor
        :param values_by_shift: shift: Shape -> values: Tensor.
        shift only contains only non-zero shift dims.
        Missing dims are interpreted as independent.
        :param shape: shape of this tensor
        """
        self.source = source
        self.val = simplify_add(values_by_shift)
        self._shape = shape

    def native(self, order=None):
        """
        Evaluates the value of the linear operation applied to the original source tensor.

        This is done by building a sparse matrix for all dimensions that are affected by the linear operation.
        These dimensions are detected automatically during the creation of the linear operation.
        All other dimensions (independent dimensions) are combined into a single batch dimensions for the sparse matrix multiplication.
        """
        result = self.apply(self.source)
        result_order = order if order is not None else self._shape.names
        return result.native(result_order)

    def apply(self, value: Tensor) -> NativeTensor:
        assert value.shape == self.source.shape
        mat = self.build_sparse_coordinate_matrix()
        independent_dims = self.independent_dims
        # TODO slice for missing dimensions
        order_src = value.shape.only(independent_dims).extend(value.shape.without(independent_dims))
        order_out = self._shape.only(independent_dims).extend(self._shape.without(independent_dims))
        native_src = value.native(order=order_src.names)
        native_src = native_math.reshape(native_src, (order_src.only(independent_dims).volume, order_src.without(independent_dims).volume))
        native_out = native_math.matmul(mat, native_src)
        native_out = native_math.reshape(native_out, order_out.sizes)
        return NativeTensor(native_out, order_out)

    def build_sparse_coordinate_matrix(self):
        """
        Builds a sparse matrix that represents this linear operation.
        Independent dimensions, those that can be treated as batch dimensions, are recognized automatically and ignored.

        :return: native sparse tensor
        """
        independent_dims = self.independent_dims
        out_shape = self._shape.without(independent_dims)
        src_shape = self.source.shape.without(independent_dims)
        cols = []
        vals = []
        for shift, values in self.val.items():
            cells = list(np.unravel_index(np.arange(out_shape.volume), out_shape.sizes))
            for missing_dim in src_shape.without(self._shape).names:
                cells.insert(self.source.shape.index(missing_dim), np.zeros_like(cells[0]))
            cells = [(cell + shift.get_size(dim) if dim in shift else cell) % src_shape.get_size(dim) for dim, cell in zip(src_shape.names, cells)]  # shift & wrap
            # inside = [(cell >= 0) & (cell < src_shape.get_size(dim)) for dim, cell in zip(src_shape.names, cells)]
            # inside = np.argwhere(np.all(inside, axis=0))[:, 0]
            # TODO crop if output > source
            src_indices = np.ravel_multi_index(cells, src_shape.sizes)
            cols.append(src_indices)
            vals.append(values.native(order=out_shape.names))
        cols = np.stack(cols, -1).flatten()
        vals = native_math.flatten(native_math.stack(vals, -1))
        rows = np.arange(out_shape.volume * len(self.val)) // len(self.val)
        # TODO sort indices
        return native_math.sparse_tensor((rows, cols), vals, (out_shape.volume, src_shape.volume))

    def build_sparse_csr_matrix(self):
        raise NotImplementedError()

    @property
    def dependent_dims(self):
        return reduce(Shape.combined, [t.shape for t in self.val.values()], EMPTY_SHAPE).names

    @property
    def independent_dims(self):
        return self.source.shape.without(self.dependent_dims).names

    @property
    def dtype(self):
        return self.source.dtype

    @property
    def shape(self):
        return self._shape

    def _with_shape_replaced(self, new_shape):
        raise NotImplementedError()

    def _getitem(self, selection: dict):
        starts = {dim: (item.start or 0) if isinstance(item, slice) else item for dim, item in selection.items()}
        new_shape = math.zeros(self._shape)[selection].shape
        return self.shift(starts, lambda v: v[selection], new_shape)

    def shift(self, shifts: dict, val_fun, new_shape):
        val = {}
        for shift, values in self.val.items():
            assert isinstance(shift, Shape)
            for dim, delta in reversed(shifts.items()):
                if dim not in values.shape:
                    values = math._expand(values, self._shape.get_size(dim), dim, self._shape.get_type(dim))  # dim order may be scrambled
                if delta:
                    shift = shift.with_size(dim, shift.get_size(dim) + delta) if dim in shift else shift.expand_spatial(delta, dim)
            val[shift] = val_fun(values)
        return ShiftLinOp(self.source, val, new_shape)

    def unstack(self, dimension):
        raise NotImplementedError()

    def __neg__(self):
        return ShiftLinOp(self.source, {shift: -values for shift, values in self.val.items()}, self._shape)

    def _op1(self, native_function):  # only __neg__ is linear
        raise NotImplementedError('Only linear operations are supported')

    def _op2(self, other, native_function, handle_special=False):
        if isinstance(other, ShiftLinOp):
            assert self.source is other.source
            assert self._shape == other._shape
            values = {}
            for dim_shift in self.val.keys():
                if dim_shift in other.val:
                    values[dim_shift] = self.val[dim_shift]._op2(other.val[dim_shift], native_function)
                else:
                    values[dim_shift] = self.val[dim_shift]._op2(math.zeros_like(self.val[dim_shift]), native_function)
            for dim_shift, other_values in other.val.items():
                if dim_shift not in self.val:
                    values[dim_shift] = math.zeros_like(other_values)._op2(other_values, native_function)
            return ShiftLinOp(self.source, values, self._shape)
        else:
            other = self._tensor(other)
            values = {}
            for dim_shift, val in self.val.items():
                val_, other_ = math.join_spaces(val, other)
                values[dim_shift] = val_._op2(other_, native_function)
            return ShiftLinOp(self.source, values, self._shape & other.shape)


class SparseLinearOperation(Tensor):

    def __init__(self, source: Tensor, dependency_matrix, shape):
        self.source = source
        self.dependency_matrix = dependency_matrix
        self._shape = shape

    def native(self, order=None):
        native_source = native_math.reshape(self.source.native(), (self.source.shape.batch.volume, self.source.shape.non_batch.volume))
        native = native_math.matmul(self.dependency_matrix, native_source)
        new_shape = self.source.shape.batch.combined(self._shape)
        native = native_math.reshape(native, new_shape.sizes)
        return NativeTensor(native, new_shape).native(order)

    @property
    def dtype(self):
        return self.source.dtype

    @property
    def shape(self):
        return self._shape

    def _with_shape_replaced(self, new_shape):
        raise NotImplementedError()

    def _getitem(self, selection: dict):
        indices = NativeTensor(native_math.reshape(native_math.range(self.shape.volume), self.shape.sizes), self.shape)
        selected_indices = indices[selection]
        selected_indices_native = native_math.flatten(selected_indices.native())
        selected_deps = native_math.gather(self.dependency_matrix, (selected_indices_native, slice(None)))
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
                    flat = native_math.flatten(other.native(broadcast_shape.names))
                    vertical = native_math.expand_dims(flat, -1)
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


def lin_placeholder(value: Tensor, format='shift', broadcast_dims=()) -> Tensor:
    """
    Create a placeholder tensor that can be used to track linear operations and construct a matrix to represent them efficiently.

    :param value: source tensor
    :param format: 'shift' or 'sparse'
    :param broadcast_dims: list of dimension names that are ignored.
    All values along these dimensions are expected to share the same linear operation.
    :return: placeholder tensor matching the values of `value`
    """
    tracking_shape = value.shape.without(broadcast_dims)
    if format == 'shift':
        return ShiftLinOp(value, {EMPTY_SHAPE: math.ones(EMPTY_SHAPE, value.dtype)}, value.shape)
    elif format == 'sparse':
        idx = native_math.range(tracking_shape.volume)
        ones = native_math.ones_like(idx)
        sparse_diag = choose_backend(value.native()).sparse_tensor([idx, idx], ones, shape=(tracking_shape.volume,) * 2)
        return SparseLinearOperation(value, sparse_diag, tracking_shape)
    else:
        raise NotImplementedError(format)


def sum_operators(operators):
    for o in operators[1:]:
        assert isinstance(o, (SparseLinearOperation, ShiftLinOp))
        assert o.source is operators[0].source
        assert o.shape == operators[0].shape
    if isinstance(operators[0], SparseLinearOperation):
        new_matrix = math.sum([o.dependency_matrix for o in operators], axis=0)
        return SparseLinearOperation(operators[0].source, new_matrix, operators[0].shape)
