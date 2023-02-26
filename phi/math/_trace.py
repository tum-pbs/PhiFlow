from collections import namedtuple
from typing import Callable, Dict, Set, Tuple

import numpy
import numpy as np

from .backend import choose_backend, NUMPY, Backend
from ._shape import Shape, parse_dim_order, merge_shapes, spatial, instance, batch, concat_shapes, EMPTY_SHAPE, dual, channel, non_batch
from ._magic_ops import stack, expand
from ._tensors import Tensor, wrap, disassemble_tree, disassemble_tensors, assemble_tree, CollapsedTensor, TensorStack
from ._sparse import SparseCoordinateTensor
from . import _ops as math


TracerSource = namedtuple('TracerSource', ['shape', 'dtype', 'name', 'index'])


class ShiftLinTracer(Tensor):
    """
    Tracer object for linear and affine functions.
    The sparsity pattern is assumed equal for all grid cells and is reflected in `val` (e.g. for a 5-point stencil, `val` has 5 items).
    The Tensors stored in `val` include position-dependent dimensions, allowing for different stencils at different positions.
    Dimensions not contained in any `val` Tensor are treated as independent (batch dimensions).
    """

    def __init__(self, source: TracerSource, values_by_shift: dict, shape: Shape, bias: Tensor):
        """
        Args:
            source: placeholder tensor
            values_by_shift: `dict` mapping relative shifts (`Shape`) to value Tensors.
                Shape keys only contain non-zero shift dims. Missing dims are interpreted as independent.
            shape: shape of this tensor
            bias: Constant Tensor to be added to the multiplication output, A*x + b.
                A bias naturally arises at boundary cells with non-trivial boundary conditions if no ghost cells are added to the matrix.
                When non-zero, this tracer technically represents an affine function, not a linear one.
                However, the bias can be subtracted from the solution vector when solving a linear system, allowing this function to be solved with regular linear system solvers.
        """
        assert isinstance(source, TracerSource)
        self.source = source
        self.val: Dict[Shape, Tensor] = simplify_add(values_by_shift)
        for shift_ in self.val.keys():
            assert shift_.only(sorted(shift_.names), reorder=True) == shift_
        self.bias = bias
        self._shape = shape

    def __repr__(self):
        return f"Linear tracer {self._shape}"

    def native(self, order: str or tuple or list or Shape = None):
        """
        Evaluates the value of the linear operation applied to the original source tensor.

        This is done by building a sparse matrix for all dimensions that are affected by the linear operation.
        These dimensions are detected automatically during the creation of the linear operation.
        All other dimensions (independent dimensions) are combined into a single batch dimensions for the sparse matrix multiplication.

        Args:
          order: str or tuple or list:  (Default value = None)

        Returns:

        """
        order = parse_dim_order(order, check_rank=self.rank)
        result = self.apply(self.source)
        result_order = order if order is not None else self._shape.names
        return result.native(result_order)

    @property
    def dependent_dims(self) -> Set[str]:
        """
        Dimensions relevant to the linear operation.
        This includes `pattern_dims` as well as dimensions along which only the values vary.
        These dimensions cannot be parallelized trivially with a non-batched matrix.
        """
        return self.pattern_dim_names | set(sum([t.shape.names for t in self.val.values()], ())) | set(self.bias.shape.names)

    @property
    def pattern_dim_names(self) -> Set[str]:
        """
        Dimensions along which the sparse matrix contains off-diagonal elements.
        These dimensions must be part of the sparse matrix and cannot be parallelized.
        """
        return set(sum([offset.names for offset in self.val], ()))

    @property
    def pattern_dims(self) -> Shape:
        return self.source.shape.only(self.pattern_dim_names)

    @property
    def dtype(self):
        return self.source.dtype

    @property
    def shape(self):
        return self._shape

    def _with_shape_replaced(self, new_shape):
        raise NotImplementedError()

    @property
    def _is_tracer(self) -> bool:
        return True

    def _getitem(self, selection: dict):
        starts = {dim: (item.start or 0) if isinstance(item, slice) else item for dim, item in selection.items()}
        new_shape = math.zeros(self._shape)[selection].shape
        return self.shift(starts, new_shape, lambda v: v[selection], lambda b: b[selection])

    def shift(self, shifts: dict,
              new_shape: Shape,
              val_fun: Callable,
              bias_fun: Callable = None):
        """
        Shifts all values of this tensor by `shifts`.
        Values shifted outside will be mapped with periodic boundary conditions when the matrix is built.

        Args:
            shifts: Offsets by dimension
            new_shape: Shape of the shifted tensor, must match the shape returned by `val_fun`.
            val_fun: Function to apply to the matrix values, may change the tensor shapes
            bias_fun: Function to apply to the bias vector, may change the tensor shape

        Returns:
            Shifted tensor, possibly with altered values.
        """
        val = {}
        for shift, values in self.val.items():
            assert isinstance(shift, Shape)
            for dim, delta in reversed(tuple(shifts.items())):
                if dim not in values.shape:
                    values = math.expand(values, self._shape.only(dim))  # dim order may be scrambled
                if delta:
                    shift = shift._replace_single_size(dim, shift.get_size(dim) + delta) if dim in shift else shift._expand(spatial(**{dim: delta}))
            val[shift.only(sorted(shift.names), reorder=True)] = val_fun(values)
        bias = bias_fun(self.bias)
        return ShiftLinTracer(self.source, val, new_shape, bias)

    def unstack(self, dimension):
        raise NotImplementedError()

    def __neg__(self):
        return ShiftLinTracer(self.source, {shift: -values for shift, values in self.val.items()}, self._shape, -self.bias)

    def _op1(self, native_function):
        # __neg__ is the only proper linear op1 and is implemented above.
        if native_function.__name__ == 'isfinite':
            test_output = self.apply(math.ones(self.source.shape, dtype=self.source.dtype))
            return math.is_finite(test_output)
        else:
            raise NotImplementedError('Only linear operations are supported')

    def _op2(self, other: Tensor,
             operator: Callable,
             native_function: Callable,
             op_name: str = 'unknown',
             op_symbol: str = '?') -> 'ShiftLinTracer':
        """
        Tensor-tensor operation.

        Args:
            other:
            operator:
            native_function:
        """
        assert op_symbol in '+-*/', f"Unsupported operation encountered while tracing linear function: {native_function}"
        zeros_for_missing_self = op_name not in ['add', 'radd', 'rsub']  # perform `operator` where `self == 0`
        zeros_for_missing_other = op_name not in ['add', 'radd', 'sub']  # perform `operator` where `other == 0`

        if isinstance(other, ShiftLinTracer):
            assert self.source is other.source, "Multiple linear tracers are not yet supported."
            assert set(self._shape) == set(other._shape), f"Tracers have different shapes: {self._shape} and {other._shape}"
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
            bias = operator(self.bias, other.bias)
            return ShiftLinTracer(self.source, values, self._shape, bias)
        else:
            other = self._tensor(other)
            if op_symbol in '*/':
                values = {}
                for dim_shift, val in self.val.items():
                    values[dim_shift] = operator(val, other)
                bias = operator(self.bias, other)
                return ShiftLinTracer(self.source, values, self._shape & other.shape, bias)
            elif op_symbol in '+-':
                bias = operator(self.bias, other)
                return ShiftLinTracer(self.source, self.val, self._shape & other.shape, bias)
            else:
                raise ValueError(f"Unsupported operation encountered while tracing linear function: {native_function}")

    def _natives(self) -> tuple:
        """
        This function should only be used to determine the compatible backends, this tensor should be regarded as not available.
        """
        return sum([v._natives() for v in self.val.values()], ()) + self.bias._natives()

    def _spec_dict(self) -> dict:
        raise LinearTraceInProgress(self)


class LinearTraceInProgress(Exception):

    def __init__(self, tracer: ShiftLinTracer):
        self.tracer = tracer


def simplify_add(val: dict) -> Dict[Shape, Tensor]:
    result = {}
    for shift, values in val.items():
        shift = shift[[i for i, size in enumerate(shift.sizes) if size != 0]]  # discard zeros
        if shift in result:
            result[shift] += values
        else:
            result[shift] = values
    return result


def matrix_from_function(f: Callable,
                         *args,
                         auxiliary_args=None,
                         auto_compress=False,
                         sparsify_batch=None,
                         separate_independent=False,  # not fully implemented, requires auto_compress=False
                         **kwargs) -> Tuple[Tensor, Tensor]:
    """
    Trace a linear function and construct a matrix.
    Depending on the functional form of `f`, the returned matrix may be dense or sparse.

    Args:
        f: Function to trace.
        *args: Arguments for `f`.
        auxiliary_args: Arguments in which the function is not linear.
            These parameters are not traced but passed on as given in `args` and `kwargs`.
        auto_compress: If `True`, returns a compressed matrix if supported by the backend.
        sparsify_batch: If `False`, the matrix will be batched.
            If `True`, will create dual dimensions for the involved batch dimensions.
            This will result in one large matrix instead of a batch of matrices.
        **kwargs: Keyword arguments for `f`.

    Returns:
        matrix: Matrix representing the linear dependency of the output `f` on the input of `f`.
            Input dimensions will be `dual` dimensions of the matrix while output dimensions will be regular.
        bias: Bias for affine functions or zero-vector if the function is purely linear.
    """
    assert isinstance(auxiliary_args, str) or auxiliary_args is None, f"auxiliary_args must be a comma-separated str but got {auxiliary_args}"
    from ._functional import function_parameters, f_name
    f_params = function_parameters(f)
    aux = set(s.strip() for s in auxiliary_args.split(',') if s.strip()) if isinstance(auxiliary_args, str) else f_params[1:]
    all_args = {**kwargs, **{f_params[i]: v for i, v in enumerate(args)}}
    aux_args = {k: v for k, v in all_args.items() if k in aux}
    trace_args = {k: v for k, v in all_args.items() if k not in aux}
    tree, tensors = disassemble_tree(trace_args)
    # tracing = not math.all_available(*tensors)
    natives, shapes, native_dims = disassemble_tensors(tensors, expand=False)
    # --- Trace function ---
    with NUMPY:
        src = TracerSource(tensors[0].shape, tensors[0].dtype, tuple(trace_args.keys())[0], 0)
        tracer = ShiftLinTracer(src, {EMPTY_SHAPE: math.ones()}, tensors[0].shape, math.zeros(tensors[0].shape, dtype=tensors[0].dtype))
        x_kwargs = assemble_tree(tree, [tracer])
        result = f(**x_kwargs, **aux_args)
    _, result_tensors = disassemble_tree(result)
    assert len(result_tensors) == 1, f"Linear function output must be or contain a single Tensor but got {result}"
    tracer = result_tensors[0]._simplify()
    assert tracer._is_tracer, f"Tracing linear function '{f_name(f)}' failed. Make sure only linear operations are used. Output: {tracer.shape}"
    # --- Convert to COO ---
    if sparsify_batch is None:
        if auto_compress:
            sparsify_batch = not tracer.default_backend.supports(Backend.csr_matrix_batched)
        else:
            sparsify_batch = not tracer.default_backend.supports(Backend.sparse_coo_tensor_batched)
    matrix, bias = tracer_to_coo(tracer, sparsify_batch, separate_independent)
    # --- Compress ---
    if not auto_compress:
        return matrix, bias
    if matrix.default_backend.supports(Backend.mul_csr_dense):
        return matrix.compress_rows(), bias
    # elif backend.supports(Backend.mul_csc_dense):
    #     return matrix.compress_cols(), tracer.bias
    else:
        return matrix, bias
    

def tracer_to_coo(tracer: Tensor, sparsify_batch: bool, separate_independent: bool):
    if isinstance(tracer, CollapsedTensor):
        tracer = tracer._cached if tracer.is_cached else tracer._inner  # ignore collapsed dimensions. Alternatively, we could expand the result
        return tracer_to_coo(tracer, sparsify_batch, separate_independent)
    elif isinstance(tracer, TensorStack):  # This indicates separable solves
        matrices, biases = zip(*[tracer_to_coo(t, sparsify_batch, separate_independent) for t in tracer._tensors])
        bias = stack(biases, tracer._stack_dim)
        if not separate_independent:
            indices = [math.concat_tensor([m._indices, expand(i, instance(m._indices), channel(vector=tracer._stack_dim.name))], 'vector') for i, m in enumerate(matrices)]
            indices = math.concat_tensor(indices, 'entries')
            values = math.concat_tensor([m._values for m in matrices], 'entries')
            # matrix = stack(matrices, tracer._stack_dim)
            dense_shape = concat_shapes(matrices[0]._dense_shape, tracer._stack_dim)
            matrix = SparseCoordinateTensor(indices, values, dense_shape, can_contain_double_entries=False, indices_sorted=False)
        else:
            matrix = stack(matrices, tracer._stack_dim)
        return matrix, bias
    elif not tracer._is_tracer:  # This part of the output is independent of the input
        return expand(0, tracer.shape), tracer
    assert isinstance(tracer, ShiftLinTracer), f"Tracing linear function returned an unsupported construct: {type(tracer)}"
    assert batch(tracer.pattern_dims).is_empty, f"Batch dimensions may not be sliced in linear operations but got pattern for {batch(tracer.pattern_dims)}"
    missing_dims = tracer.source.shape.without(tracer.shape)  # these were sliced off
    ignored_dims = tracer.source.shape.without(tracer.shape.only(tracer.dependent_dims) if sparsify_batch else tracer.pattern_dim_names).without(missing_dims)  # these will be parallelized and not added to the matrix
    out_shape = tracer.shape.without(ignored_dims)
    typed_src_shape = tracer.source.shape.without(ignored_dims)
    src_shape = dual(**typed_src_shape.untyped_dict)
    sliced_src_shape = src_shape.without(dual(**missing_dims.untyped_dict))
    batch_val = merge_shapes(*tracer.val.values()).without(out_shape)
    if non_batch(out_shape).is_empty:
        assert len(tracer.val) == 1 and non_batch(tracer.val[EMPTY_SHAPE]) == EMPTY_SHAPE
        return tracer.val[EMPTY_SHAPE], tracer.bias
    out_indices = []
    src_indices = []
    values = []
    for shift_, shift_val in tracer.val.items():
        if shift_val.default_backend is NUMPY:  # sparsify stencil further
            native_shift_values = math.reshaped_native(shift_val, [batch_val, *out_shape], force_expand=True)
            mask = np.sum(abs(native_shift_values), 0)  # only 0 where no batch entry has a non-zero value
            out_idx = numpy.nonzero(mask)
            src_idx = [(component + shift_.get_size(dim)) % typed_src_shape.get_size(dim) if dim in shift_ else component for component, dim in zip(out_idx, out_shape)]
            values.append(native_shift_values[(slice(None), *out_idx)])
        else:  # add full stencil tensor
            out_idx = np.unravel_index(np.arange(out_shape.volume), out_shape.sizes) if out_shape else 0
            src_idx = [(component + shift_.get_size(dim)) % typed_src_shape.get_size(dim) if dim in shift_ else component for component, dim in zip(out_idx, out_shape)]
            values.append(math.reshaped_native(shift_val, [batch_val, out_shape], force_expand=True))
        out_indices.append(out_idx)
        src_idx_all = []
        for dim in typed_src_shape:
            if dim in missing_dims:
                if not separate_independent:
                    offset = shift_.get_size(dim, default=0)
                    src_idx_all.append(np.zeros_like(src_idx[0]) + offset)
            else:
                src_idx_all.append(src_idx[out_shape.index(dim)])
        src_indices.append(src_idx_all)
    indices_np = np.concatenate([np.concatenate(src_indices, axis=1), np.concatenate(out_indices, axis=1)]).T
    indices = wrap(indices_np, instance('entries'), channel(vector=(sliced_src_shape if separate_independent else src_shape).names + out_shape.names))
    backend = choose_backend(*values)
    values = math.reshaped_tensor(backend.concat(values, axis=-1), [batch_val, instance('entries')])
    dense_shape = concat_shapes((sliced_src_shape if separate_independent else src_shape) & out_shape)
    matrix = SparseCoordinateTensor(indices, values, dense_shape, can_contain_double_entries=False, indices_sorted=False)
    return matrix, tracer.bias
