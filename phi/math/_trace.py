from functools import reduce
from typing import Tuple, Callable

import numpy as np

from .backend import choose_backend
from ._shape import EMPTY_SHAPE, Shape, shape, parse_dim_order, SPATIAL_DIM
from ._tensors import Tensor, NativeTensor, TensorStack, CollapsedTensor
from . import _functions as math


def simplify_add(val: dict):
    result = {}
    for shift, values in val.items():
        shift = shift[[i for i, size in enumerate(shift.sizes) if size != 0]]  # discard zeros
        if shift in result:
            result[shift] += values
        else:
            result[shift] = values
    return result


class ShiftLinOp(Tensor):

    def __init__(self, source: Tensor, values_by_shift: dict, shape: Shape):
        """


        Args:
          source: placeholder tensor
          values_by_shift: shift: Shape -> values: Tensor.
        shift only contains only non-zero shift dims.
        Missing dims are interpreted as independent.
          shape: shape of this tensor
        """
        self.source = source
        self.val = simplify_add(values_by_shift)
        self._shape = shape

    def native(self, order: str or tuple or list = None):
        """
        Evaluates the value of the linear operation applied to the original source tensor.
        
        This is done by building a sparse matrix for all dimensions that are affected by the linear operation.
        These dimensions are detected automatically during the creation of the linear operation.
        All other dimensions (independent dimensions) are combined into a single batch dimensions for the sparse matrix multiplication.

        Args:
          order: str or tuple or list:  (Default value = None)

        Returns:

        """
        order = parse_dim_order(order)
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
        backend = choose_backend(native_src)
        native_src = backend.reshape(native_src, (order_src.only(independent_dims).volume, order_src.without(independent_dims).volume))
        native_out = backend.matmul(mat, native_src)
        native_out = backend.reshape(native_out, order_out.sizes)
        return NativeTensor(native_out, order_out)

    def build_sparse_coordinate_matrix(self):
        """
        Builds a sparse matrix that represents this linear operation.
        Independent dimensions, those that can be treated as batch dimensions, are recognized automatically and ignored.
        
        :return: native sparse tensor

        Args:

        Returns:

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
            src_indices = np.ravel_multi_index(cells, src_shape.sizes)
            cols.append(src_indices)
            vals.append(CollapsedTensor(values, out_shape).native())
        cols = np.stack(cols, -1).flatten()
        backend = choose_backend(*vals)
        vals = backend.flatten(backend.stack(vals, -1))
        rows = np.arange(out_shape.volume * len(self.val)) // len(self.val)
        # TODO sort indices
        return choose_backend(rows, cols, vals).sparse_tensor((rows, cols), vals, (out_shape.volume, src_shape.volume))

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

    @property
    def _is_special(self) -> bool:
        return True

    def _getitem(self, selection: dict):
        starts = {dim: (item.start or 0) if isinstance(item, slice) else item for dim, item in selection.items()}
        new_shape = math.zeros(self._shape)[selection].shape
        return self.shift(starts, lambda v: v[selection], new_shape)

    def shift(self, shifts: dict, val_fun, new_shape):
        """
        Shifts all values of this tensor by `shifts`.
        Values shifted outside will be mapped with periodic boundary conditions when the matrix is built, see `build_sparse_coordinate_matrix()`.

        Args:
            shifts: Offsets by dimension
            val_fun: Function to apply to the matrix values, may change the tensor shapes
            new_shape: Shape of the shifted tensor, must match the shape returned by `val_fun`.

        Returns:
            Shifted tensor, possibly with altered values.
        """
        val = {}
        for shift, values in self.val.items():
            assert isinstance(shift, Shape)
            for dim, delta in reversed(tuple(shifts.items())):
                if dim not in values.shape:
                    values = math._expand_dims(values, self._shape.only(dim))  # dim order may be scrambled
                if delta:
                    shift = shift.with_size(dim, shift.get_size(dim) + delta) if dim in shift else shift.expand(delta, dim, SPATIAL_DIM)
            val[shift] = val_fun(values)
        return ShiftLinOp(self.source, val, new_shape)

    def unstack(self, dimension):
        raise NotImplementedError()

    def __neg__(self):
        return ShiftLinOp(self.source, {shift: -values for shift, values in self.val.items()}, self._shape)

    def _op1(self, native_function):  # only __neg__ is linear
        raise NotImplementedError('Only linear operations are supported')

    def __add__(self, other):
        return self._op2(other, lambda x, y: x + y, lambda x, y: choose_backend(x, y).add(x, y), zeros_for_missing_self=False, zeros_for_missing_other=False)

    def __sub__(self, other):
        return self._op2(other, lambda x, y: x - y, lambda x, y: choose_backend(x, y).sub(x, y), zeros_for_missing_other=False)

    def __rsub__(self, other):
        return self._op2(other, lambda x, y: y - x, lambda x, y: choose_backend(x, y).sub(y, x), zeros_for_missing_self=False)

    def _op2(self, other: Tensor,
             operator: Callable,
             native_function: Callable,
             zeros_for_missing_self=True,
             zeros_for_missing_other=True) -> 'ShiftLinOp':
        """
        Tensor-tensor operation.

        Args:
            other:
            operator:
            native_function:
            zeros_for_missing_self: perform `operator` where `self == 0`
            zeros_for_missing_other: perform `operator` where `other == 0`
        """
        if isinstance(other, ShiftLinOp):
            assert self.source is other.source
            assert self._shape == other._shape
            values = {}
            for dim_shift in self.val.keys():
                if dim_shift in other.val:
                    values[dim_shift] = operator(self.val[dim_shift], other.val[dim_shift])
                else:
                    if zeros_for_missing_other:
                        values[dim_shift] = operator(self.val[dim_shift], math.zeros_like(self.val[dim_shift]))
                    else:
                        values[dim_shift] = self.val[dim_shift]
            for dim_shift, other_values in other.val.items():
                if dim_shift not in self.val:
                    if zeros_for_missing_self:
                        values[dim_shift] = operator(math.zeros_like(other_values), other_values)
                    else:
                        values[dim_shift] = other_values
            return ShiftLinOp(self.source, values, self._shape)
        else:
            other = self._tensor(other)
            values = {}
            for dim_shift, val in self.val.items():
                val_, other_ = math.join_spaces(val, other)
                values[dim_shift] = operator(val_, other_)
            return ShiftLinOp(self.source, values, self._shape & other.shape)

    def __tensor_reduce__(self,
                dims: Tuple[str],
                native_function: Callable,
                collapsed_function: Callable = lambda inner_reduced, collapsed_dims_to_reduce: inner_reduced,
                unaffected_function: Callable = lambda value: value):
        if all(dim not in self._shape for dim in dims):
            return unaffected_function(self)
        raise NotImplementedError()

    def _natives(self) -> tuple:
        raise NotImplementedError()  # should not be used, this tensor should be regarded as not available
        # return (self.source._natives(),) + sum([v._natives() for v in self.val.values()], ())


def lin_placeholder(value: Tensor) -> Tensor:
    """
    Create a placeholder tensor that can be used to trace linear operations and construct a matrix to represent them efficiently.

    Args:
        value: source tensor
        format: shift' or 'sparse' (Default value = 'shift')
        broadcast_dims: list of dimension names that are ignored.
            All values along these dimensions are expected to share the same linear operation.

    Returns:
        Placeholder tensor matching the values of `value`
    """
    return ShiftLinOp(value, {EMPTY_SHAPE: math.ones(EMPTY_SHAPE, value.dtype)}, value.shape)
