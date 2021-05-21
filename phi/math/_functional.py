import time
import uuid
import warnings
from copy import copy
from functools import reduce
from typing import Tuple, Callable, Dict, Generic, List, TypeVar, Any

import numpy as np

from . import _ops as math
from ._ops import choose_backend_t, zeros_like, all_available, print_, reshaped_native, reshaped_tensor, batch_stack
from ._shape import EMPTY_SHAPE, Shape, parse_dim_order, SPATIAL_DIM, shape, vector_add, combine_safe
from ._tensors import Tensor, NativeTensor, CollapsedTensor, disassemble_nested, TensorLike, assemble_nested, copy_with, \
    disassemble_tensors, assemble_tensors, TensorLikeType, variable_attributes, wrap
from .backend import choose_backend, Backend, get_current_profile, get_precision
from .backend._backend import BasicSolveResult, FullSolveResult

X = TypeVar('X')
Y = TypeVar('Y')


class SignatureKey:

    def __init__(self,
                 source_function: Callable or None,
                 nest,
                 shapes: Shape or Tuple[Shape],
                 kwargs: dict or None,
                 backend: Backend):
        assert isinstance(nest, TensorLike), nest
        if source_function is None:  # this is an input signature
            assert isinstance(shapes, tuple)
        self.source_function = source_function
        self.nest = nest
        self.shapes = shapes
        self.kwargs = kwargs
        self.backend = backend

    def __repr__(self):
        return f"{self.nest} with shapes {self.shapes}"

    def __eq__(self, other: 'SignatureKey'):
        assert isinstance(other, SignatureKey)
        return self.nest == other.nest and self.shapes == other.shapes and self.kwargs == other.kwargs and self.backend == other.backend

    def __hash__(self):
        return hash(self.shapes) + hash(self.backend)

    def matches_structure_and_names(self, other: 'SignatureKey'):
        assert isinstance(other, SignatureKey)
        return self.nest == other.nest and all(s1.names == s2.names for s1, s2 in zip(self.shapes, other.shapes)) and self.kwargs == other.kwargs and self.backend == other.backend

    def extrapolate(self, rec_in: 'SignatureKey', new_in: 'SignatureKey') -> 'SignatureKey':
        assert self.source_function is not None, "extrapolate() must be called on output keys"
        shapes = [self._extrapolate_shape(s, rec_in, new_in) for s in self.shapes]
        return SignatureKey(self.source_function, self.nest, shapes, self.kwargs, self.backend)

    @staticmethod
    def _extrapolate_shape(shape_: Shape, rec_in: 'SignatureKey', new_in: 'SignatureKey') -> Shape:
        sizes = []
        for dim, size in shape_.named_sizes:
            for p_in, n_in in zip(rec_in.shapes, new_in.shapes):
                if dim in p_in and size == p_in.get_size(dim):
                    sizes.append(n_in.get_size(dim))
                    break
            else:
                raise ValueError(shape_, rec_in, new_in)
        return shape_.with_sizes(sizes)


def match_output_signature(new_in: SignatureKey, recorded_mappings: Dict[SignatureKey, SignatureKey]) -> SignatureKey:
    for rec_in, rec_out in recorded_mappings.items():
        if rec_in == new_in:  # exact match
            return rec_out
    for rec_in, rec_out in recorded_mappings.items():
        if rec_in.matches_structure_and_names(new_in):
            return rec_out.extrapolate(rec_in, new_in)
    raise KeyError(f"Not output shape found for input shapes {new_in}. "
                   f"Maybe the backend extrapolated the concrete function from another trace? "
                   f"Registered transforms: {recorded_mappings}")


def key_from_args(*args, **kwargs):
    nest, tensors = disassemble_nested(args)
    backend = math.choose_backend_t(*tensors)
    natives, shapes = disassemble_tensors(tensors)
    key = SignatureKey(None, nest, shapes, kwargs, backend)
    return key, natives


class JitFunction:

    def __init__(self, f: Callable):
        self.f = f
        self.traces: Dict[SignatureKey, Callable] = {}
        self.recorded_mappings: Dict[SignatureKey, SignatureKey] = {}

    def _jit_compile(self, in_key: SignatureKey):
        def f_native(*natives, **kwargs):
            assert not kwargs
            in_tensors = assemble_tensors(natives, in_key.shapes)
            values = assemble_nested(in_key.nest, in_tensors)
            assert isinstance(values, tuple)  # was disassembled from *args
            result = self.f(*values, **in_key.kwargs)  # Tensor or tuple/list of Tensors
            nest, out_tensors = disassemble_nested(result)
            result_natives, result_shapes = disassemble_tensors(out_tensors)
            self.recorded_mappings[in_key] = SignatureKey(f_native, nest, result_shapes, None, in_key.backend)
            return result_natives
        return in_key.backend.jit_compile(f_native)

    def __call__(self, *args, **kwargs):
        key, natives = key_from_args(*args, **kwargs)
        if not key.backend.supports(Backend.jit_compile):
            warnings.warn(f"jit_copmile() not supported by {key.backend}. Running function '{self.f.__name__}' as-is.")
            return self.f(*args, **kwargs)
        if key not in self.traces:
            self.traces[key] = self._jit_compile(key)
        native_result = self.traces[key](*natives)
        output_key = match_output_signature(key, self.recorded_mappings)
        output_tensors = assemble_tensors(native_result, output_key.shapes)
        return assemble_nested(output_key.nest, output_tensors)

    def __repr__(self):
        return f"jit({self.f.__name__})"


def jit_compile(f: Callable) -> Callable:
    """
    Compiles a graph based on the function `f`.
    The graph compilation is performed just-in-time (jit) when the returned function is called for the first time.

    The traced function will compute the same result as `f` but may run much faster.
    Some checks may be disabled in the compiled function.

    Can be used as a decorator:
    ```python
    @math.jit_compile
    def my_function(x: math.Tensor) -> math.Tensor:
    ```

    Compilation is implemented for the following backends:

    * PyTorch: [`torch.jit.trace`](https://pytorch.org/docs/stable/jit.html)
    * TensorFlow: [`tf.function`](https://www.tensorflow.org/guide/function)
    * Jax: [`jax.jit`](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#using-jit-to-speed-up-functions)

    See Also:
        `jit_compile_linear()`

    Args:
        f: Function to be traced.
            All arguments must be of type `Tensor` returning a single `Tensor` or a `tuple` or `list` of tensors.

    Returns:
        Function with similar signature and return values as `f`.
    """
    return f if isinstance(f, (JitFunction, LinearFunction)) else JitFunction(f)


class LinearFunction(Generic[X, Y], Callable[[X], Y]):
    """
    Just-in-time compiled linear function of `Tensor` arguments and return values.

    Use `jit_compile_linear()` to create a linear function representation.
    """

    def __init__(self, f):
        self.f = f
        self.tracers: Dict[SignatureKey, ShiftLinTracer] = {}
        self.nl_jit = JitFunction(f)  # for backends that do not support sparse matrices

    def _trace(self, in_key: SignatureKey) -> 'ShiftLinTracer':
        assert len(in_key.shapes) == 1, "Linear functions only support one argument."
        with in_key.backend:
            x = math.ones(in_key.shapes[0])
            tracer = ShiftLinTracer(x, {EMPTY_SHAPE: math.ones()}, x.shape)
        f_input = assemble_nested(in_key.nest, [tracer])
        assert isinstance(f_input, tuple)
        result = self.f(*f_input)
        _, result_tensors = disassemble_nested(result)
        assert len(result_tensors) == 1, f"Linear function must return a single Tensor or tensor-like but got {result}"
        result_tensor = result_tensors[0]
        assert isinstance(result_tensor, ShiftLinTracer), f"Tracing linear function '{self.f.__name__}' failed. Make sure only linear operations are used."
        return result_tensor

    def __call__(self, *args: X, **kwargs) -> Y:
        nest, tensors = disassemble_nested(args)
        assert tensors, "Linear function requires at least one argument"
        if any(isinstance(t, ShiftLinTracer) for t in tensors):
            # TODO: if t is identity, use cached ShiftLinTracer, otherwise multiply two ShiftLinTracers
            return self.f(*args, **kwargs)
        backend = math.choose_backend_t(*tensors)
        if not backend.supports(Backend.sparse_tensor):
            warnings.warn(f"Sparse matrices are not supported by {backend}. Falling back to regular jit compilation.")
            return self.nl_jit(*args, **kwargs)
        natives, shapes = disassemble_tensors(tensors)
        key = SignatureKey(None, nest, shapes, kwargs, backend)
        if key not in self.tracers:
            self.tracers[key] = self._trace(key)
        return self.tracers[key].apply(tensors[0])

    def sparse_coordinate_matrix(self, *args: Tensor, **kwargs):
        key, _ = key_from_args(*args, **kwargs)
        assert key.backend.supports(Backend.sparse_tensor)
        if key not in self.tracers:
            self.tracers[key] = self._trace(key)
        return self.tracers[key].get_sparse_coordinate_matrix()

    def stencil_inspector(self, *args, **kwargs):
        key, _ = key_from_args(*args, **kwargs)
        tracer = self._trace(key)

        def print_stencil(**indices):
            pos = shape(**indices)
            print(f"{self.f.__name__}: {pos} = {' + '.join(f'{val[indices]} * {vector_add(pos, offset)}' for offset, val in tracer.val.items() if (val[indices] != 0).all)}")

        return print_stencil


def jit_compile_linear(f: Callable[[X], Y]) -> 'LinearFunction[X, Y]':
    """
    Compile an optimized representation of the linear function `f`.

    Can be used as a decorator:

    ```python
    @math.jit_compile_linear
    def my_linear_function(x: math.Tensor) -> math.Tensor:
    ```

    See Also:
        `jit_compile()`

    Args:
        f: Linear function with `Tensor` positional arguments and return value(s).

    Returns:
        `LinearFunction` with similar signature and return values as `f`.
    """
    if isinstance(f, JitFunction):
        f = f.f  # cannot trace linear function from jitted version
    return f if isinstance(f, LinearFunction) else LinearFunction(f)


class GradientFunction:

    def __init__(self, f: Callable, wrt: tuple, get_output: bool):
        self.f = f
        self.wrt = wrt
        self.get_output = get_output
        self.grads: Dict[SignatureKey, Callable] = {}
        self.recorded_mappings: Dict[SignatureKey, SignatureKey] = {}

    def _trace_grad(self, in_key: SignatureKey, wrt_natives):
        def f_native(*natives, **kwargs):
            assert not kwargs
            in_tensors = assemble_tensors(natives, in_key.shapes)
            values = assemble_nested(in_key.nest, in_tensors)
            assert isinstance(values, tuple)  # was disassembled from *args
            result = self.f(*values, **in_key.kwargs)  # Tensor or tuple/list of Tensors
            nest, out_tensors = disassemble_nested(result)
            result_natives, result_shapes = disassemble_tensors(out_tensors)
            self.recorded_mappings[in_key] = SignatureKey(f_native, nest, result_shapes, None, in_key.backend)
            return result_natives
        return in_key.backend.functional_gradient(f_native, wrt=wrt_natives, get_output=self.get_output)

    def __call__(self, *args, **kwargs):
        key, natives = key_from_args(*args, **kwargs)
        if not key.backend.supports(Backend.functional_gradient):
            if math.default_backend().supports(Backend.functional_gradient):
                warnings.warn(f"Using {math.default_backend()} for gradient computation because {key.backend} does not support functional_gradient()")
                key.backend = math.default_backend()
            else:
                raise AssertionError(f"functional_gradient() not supported by {key.backend}.")
        wrt_tensors = self._track_wrt(args)
        wrt_natives = self._track_wrt_natives(wrt_tensors, disassemble_nested(args)[1])
        if key not in self.grads:
            self.grads[key] = self._trace_grad(key, wrt_natives)
        native_result = self.grads[key](*natives)
        output_key = match_output_signature(key, self.recorded_mappings)
        if self.get_output:
            result_shapes = list(output_key.shapes) + [key.shapes[i] for i in wrt_tensors]
            output_tensors = assemble_tensors(native_result, result_shapes)
            output_structure, grad_tuple = assemble_nested((output_key.nest, [key.nest[i] for i in wrt_tensors]), output_tensors)
            return output_structure, grad_tuple
        else:
            output_tensors = assemble_tensors(native_result, [key.shapes[i] for i in wrt_tensors])
            return assemble_nested([key.nest[i] for i in wrt_tensors], output_tensors)

    def __repr__(self):
        return f"jit({self.f.__name__})"

    def _track_wrt(self, args):
        wrt_tensors = []
        for i, arg in enumerate(args):
            _, tensors = disassemble_nested(arg)
            wrt_tensors.extend([i] * len(tensors))
        return [t_i for t_i, arg_i in enumerate(wrt_tensors) if arg_i in self.wrt]

    @staticmethod
    def _track_wrt_natives(wrt_tensors, values):
        wrt_natives = []
        for i, value in enumerate(values):
            wrt_natives.extend([i] * len(value._natives()))
        return [n_i for n_i, t_i in enumerate(wrt_natives) if t_i in wrt_tensors]


def functional_gradient(f: Callable, wrt: tuple or list = (0,), get_output=True) -> Callable:
    """
    Creates a function which computes the spatial_gradient of `f`.

    Example:

    ```python
    def loss_function(x, y):
        prediction = f(x)
        loss = math.l2_loss(prediction - y)
        return loss, prediction

    dx, = functional_gradient(loss_function)(x, y)

    loss, prediction, dx, dy = functional_gradient(loss_function, wrt=(0, 1),
                                                 get_output=True)(x, y)
    ```

    Args:
        f: Function to be differentiated.
            `f` must return a floating point `Tensor` with rank zero.
            It can return additional tensors which are treated as auxiliary data and will be returned by the spatial_gradient function if `return_values=True`.
            All arguments for which the spatial_gradient is computed must be of dtype float or complex.
        get_output: Whether the spatial_gradient function should also return the return values of `f`.
        wrt: Arguments of `f` with respect to which the spatial_gradient should be computed.
            Example: `wrt_indices=[0]` computes the spatial_gradient with respect to the first argument of `f`.

    Returns:
        Function with the same arguments as `f` that returns the value of `f`, auxiliary data and spatial_gradient of `f` if `get_output=True`, else just the spatial_gradient of `f`.
    """
    return GradientFunction(f, wrt, get_output)


class CustomGradientFunction:

    def __init__(self, f: Callable, gradient: Callable):
        self.f = f
        self.gradient = gradient
        self.traces: Dict[SignatureKey, Callable] = {}
        self.recorded_mappings: Dict[SignatureKey, SignatureKey] = {}

    def _trace(self, in_key: SignatureKey):
        def f_native(*natives, **kwargs):
            assert not kwargs
            in_tensors = assemble_tensors(natives, in_key.shapes)
            values = assemble_nested(in_key.nest, in_tensors)
            assert isinstance(values, tuple)  # was disassembled from *args
            result = self.f(*values, **in_key.kwargs)  # Tensor or tuple/list of Tensors
            nest, out_tensors = disassemble_nested(result)
            result_natives, result_shapes = disassemble_tensors(out_tensors)
            self.recorded_mappings[in_key] = SignatureKey(f_native, nest, result_shapes, None, in_key.backend)
            return result_natives

        def g_native(x_natives, y_natives, dy_natives):
            out_key = self.recorded_mappings[in_key]
            del self.recorded_mappings[in_key]
            x_tensors = assemble_tensors(x_natives, in_key.shapes)
            y_tensors = assemble_tensors(y_natives, out_key.shapes)
            dy_tensors = assemble_tensors(dy_natives, out_key.shapes)
            x = assemble_nested(in_key.nest, x_tensors)
            assert isinstance(x, tuple)
            y = assemble_nested(out_key.nest, y_tensors)
            dy = assemble_nested(out_key.nest, dy_tensors)
            result = self.gradient(*x, y, dy, **in_key.kwargs)
            assert isinstance(result, (tuple, list)), "Gradient function must return tuple or list"
            result_natives = self.incomplete_nested_to_natives(result, in_key.nest, list(in_key.shapes))
            return result_natives

        return in_key.backend.custom_gradient(f_native, g_native)

    def __call__(self, *args, **kwargs):
        key, natives = key_from_args(*args, **kwargs)
        if not key.backend.supports(Backend.functional_gradient) and not key.backend.supports(Backend.gradients):
            return self.f(*args, **kwargs)  # no need to use custom gradient if gradients aren't supported anyway
        elif not key.backend.supports(Backend.custom_gradient):
            warnings.warn(f"custom_gradient() not supported by {key.backend}. Running function '{self.f.__name__}' as-is.")
            return self.f(*args, **kwargs)
        if key not in self.traces:
            self.traces[key] = self._trace(key)
        native_result = self.traces[key](*natives)
        output_key = match_output_signature(key, self.recorded_mappings)
        output_tensors = assemble_tensors(native_result, output_key.shapes)
        return assemble_nested(output_key.nest, output_tensors)

    def __repr__(self):
        return f"jit({self.f.__name__})"

    @staticmethod
    def incomplete_nested_to_natives(incomplete, nest, complete_shapes: List[Shape]) -> list:
        """ None in nest means there is a tensor. """
        if nest is None:
            c_shape = complete_shapes.pop(0)
            if incomplete is None:
                return [None] * c_shape.shape.without('dims').volume
            else:
                assert isinstance(incomplete, Tensor)
                return list(incomplete._natives())
        elif isinstance(nest, (tuple, list)):
            if incomplete is None:
                raise NotImplementedError()
            else:
                assert type(nest) == type(incomplete) and len(nest) == len(incomplete)
                natives = []
                for i_item, c_item in zip(incomplete, nest):
                    natives_item = CustomGradientFunction.incomplete_nested_to_natives(i_item, c_item, complete_shapes)
                    natives.extend(natives_item)
                return natives
        elif isinstance(nest, dict):
            raise NotImplementedError()
        elif isinstance(nest, TensorLike):
            attributes = variable_attributes(nest)
            natives = []
            for attr in attributes:
                n_val = getattr(nest, attr)
                i_val = getattr(incomplete, attr) if incomplete is not None else None
                natives_item = CustomGradientFunction.incomplete_nested_to_natives(i_val, n_val, complete_shapes)
                natives.extend(natives_item)
            return natives
        else:
            raise ValueError(f"Value must be Tensor or tensor-like but got {type(nest)}")


def custom_gradient(f: Callable, gradient: Callable):
    """
    Creates a function based on `f` that uses a custom gradient for the backpropagation pass.

    *Warning* This method can lead to memory leaks if the gradient funcion is not called.
    Make sure to pass tensors without gradients if the gradient is not required, see `stop_gradient()`.

    Args:
        f: Forward function mapping `Tensor` arguments `x` to a single `Tensor` output or sequence of tensors `y`.
        gradient: Function to compute the vector-Jacobian product for backpropropagation. Will be called as `gradient(*x, *y, *dy) -> *dx`.

    Returns:
        Function with similar signature and return values as `f`. However, the returned function does not support keyword arguments.
    """
    return CustomGradientFunction(f, gradient)


def is_tracer(t: Tensor):
    return isinstance(t, ShiftLinTracer)


def simplify_add(val: dict) -> Dict[Shape, Tensor]:
    result = {}
    for shift, values in val.items():
        shift = shift[[i for i, size in enumerate(shift.sizes) if size != 0]]  # discard zeros
        if shift in result:
            result[shift] += values
        else:
            result[shift] = values
    return result


class ShiftLinTracer(Tensor):

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
        self.val: Dict[Shape, Tensor] = simplify_add(values_by_shift)
        self._shape = shape
        self._sparse_coo = None

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
        mat = self.get_sparse_coordinate_matrix().native()
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

    def get_sparse_coordinate_matrix(self) -> 'FixedShiftSparseTensor':
        """
        Builds a sparse matrix that represents this linear operation.
        Independent dimensions, those that can be treated as batch dimensions, are recognized automatically and ignored.
        
        :return: native sparse tensor

        Args:

        Returns:

        """
        if self._sparse_coo is not None:
            return self._sparse_coo
        independent_dims = self.independent_dims
        out_shape = self._shape.without(independent_dims)
        src_shape = self.source.shape.without(independent_dims)
        cols = []
        vals = []
        for shift, values in self.val.items():
            cells = list(cell_indices(out_shape))
            for missing_dim in src_shape.without(self._shape).names:
                cells.insert(self.source.shape.index(missing_dim), np.zeros_like(cells[0]))
            cells = [(cell + shift.get_size(dim) if dim in shift else cell) % src_shape.get_size(dim) for dim, cell in zip(src_shape.names, cells)]  # shift & wrap
            src_indices = cell_number(cells, src_shape)
            cols.append(src_indices)
            vals.append(CollapsedTensor(values, out_shape).native())
        cols = np.stack(cols, -1).flatten()
        backend = choose_backend(*vals)
        vals = backend.flatten(backend.stack(vals, -1))
        rows = np.arange(out_shape.volume * len(self.val)) // len(self.val)
        # TODO sort indices?
        self._sparse_coo = FixedShiftSparseTensor((out_shape.volume, src_shape.volume),
                                                  set(self.val.keys()), rows, cols,
                                                  NativeTensor(vals, shape(nnz=len(vals))),
                                                  self.dependent_dims)
        return self._sparse_coo

    def build_sparse_csr_matrix(self):
        raise NotImplementedError()

    @property
    def dependent_dims(self):
        return reduce(Shape.combined, [t.shape for t in self.val.values()], EMPTY_SHAPE)

    @property
    def independent_dims(self):
        return self.source.shape.without(self.dependent_dims)

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
        Values shifted outside will be mapped with periodic boundary conditions when the matrix is built, see `get_sparse_coordinate_matrix()`.

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
        return ShiftLinTracer(self.source, val, new_shape)

    def unstack(self, dimension):
        raise NotImplementedError()

    def __neg__(self):
        return ShiftLinTracer(self.source, {shift: -values for shift, values in self.val.items()}, self._shape)

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
             zeros_for_missing_other=True) -> 'ShiftLinTracer':
        """
        Tensor-tensor operation.

        Args:
            other:
            operator:
            native_function:
            zeros_for_missing_self: perform `operator` where `self == 0`
            zeros_for_missing_other: perform `operator` where `other == 0`
        """
        if isinstance(other, ShiftLinTracer):
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
            return ShiftLinTracer(self.source, values, self._shape)
        else:
            other = self._tensor(other)
            values = {}
            for dim_shift, val in self.val.items():
                val_, other_ = math.join_spaces(val, other)
                values[dim_shift] = operator(val_, other_)
            return ShiftLinTracer(self.source, values, self._shape & other.shape)

    def _tensor_reduce(self,
                       dims: Tuple[str],
                       native_function: Callable,
                       collapsed_function: Callable = lambda inner_reduced, collapsed_dims_to_reduce: inner_reduced,
                       unaffected_function: Callable = lambda value: value):
        if all(dim not in self._shape for dim in dims):
            return unaffected_function(self)
        raise NotImplementedError()

    def _natives(self) -> tuple:
        return sum([v._natives() for v in self.val.values()], ())
        # raise NotImplementedError()  # should not be used, this tensor should be regarded as not available


def cell_indices(shape: Shape) -> tuple:
    if shape.rank > 0:
        return np.unravel_index(np.arange(shape.volume), shape.sizes)
    else:
        return 0,


def cell_number(cells, resolution: Shape):
    if resolution.rank > 0:
        return np.ravel_multi_index(cells, resolution.sizes)
    else:
        return 0,


class FixedShiftSparseTensor:

    def __init__(self, shape: tuple, indices_key, rows, cols, values: Tensor, src_shape: Shape):
        self.shape = shape
        self.indices_key = indices_key
        self.rows = rows
        self.cols = cols
        self.values = values
        self.src_shape = src_shape

    def __eq__(self, other):
        return isinstance(other, FixedShiftSparseTensor) and self.indices_key == other.indices_key and self.src_shape == other.src_shape

    def __variable_attrs__(self):
        return 'values',

    def native(self):
        backend = choose_backend(self.rows, self.cols, *self.values._natives())
        return backend.sparse_tensor((self.rows, self.cols), self.values.native(), self.shape)


class Solve(Generic[X, Y]):  # TODO move to phi.math._functional, put Tensors there
    """
    Specifies parameters and stopping criteria for solving a minimization problem or system of equations.
    """

    def __init__(self,
                 method: str,
                 relative_tolerance: float or Tensor,
                 absolute_tolerance: float or Tensor,
                 max_iterations: int or Tensor = 1000,
                 x0: X or Any = None,
                 suppress: tuple or list = (),
                 gradient_solve: 'Solve[Y, X]' or None = None):
        assert isinstance(method, str)
        self.method: str = method
        """ Optimization method to use. Available solvers depend on the solve function that is used to perform the solve. """
        self.relative_tolerance: Tensor = wrap(relative_tolerance)
        """ Relative tolerance for linear solves only. This must be `0` for minimization problems.
        For systems of equations *f(x)=y*, the final tolerance is `max(relative_tolerance * norm(y), absolute_tolerance)`. """
        self.absolute_tolerance: Tensor = wrap(absolute_tolerance)
        """ Absolut tolerance for optimization problems and linear solves.
        For systems of equations *f(x)=y*, the final tolerance is `max(relative_tolerance * norm(y), absolute_tolerance)`. """
        self.max_iterations: Tensor = wrap(max_iterations)
        """ Maximum number of iterations to perform before raising a `NotConverged` error is raised. """
        self.x0 = x0
        """ Initial guess for the method, of same type and dimensionality as the solve result.
         This property must be set to a value compatible with the solution `x` before running a method. """
        assert all(issubclass(err, ConvergenceException) for err in suppress)
        self.suppress: tuple = tuple(suppress)
        """ Error types to suppress; `tuple` of `ConvergenceException` types. For these errors, the solve function will instead return the partial result without raising the error. """
        self._gradient_solve: Solve[Y, X] = gradient_solve
        self.id = str(uuid.uuid4())

    @property
    def gradient_solve(self) -> 'Solve[Y, X]':
        """
        Parameters to use for the gradient pass when an implicit gradient is computed.
        If `None`, a duplicate of this `Solve` is created for the gradient solve.

        In any case, the gradient solve information will be stored in `gradient_solve.result`.
        """
        if self._gradient_solve is None:
            self._gradient_solve = copy(self)
            self._gradient_solve.x0 = None
        return self._gradient_solve

    def __repr__(self):
        return f"{self.method} with tolerance {self.relative_tolerance} (rel), {self.absolute_tolerance} (abs), max_iterations={self.max_iterations}"

    def __eq__(self, other):
        if not isinstance(other, Solve):
            return False
        if self.method != other.method \
                or self.absolute_tolerance != other.absolute_tolerance \
                or self.relative_tolerance != other.relative_tolerance \
                or self.max_iterations != other.max_iterations \
                or self.suppress != other.suppress:
            return False
        if self.x0 is None:
            return other.x0 is None
        else:
            raise AssertionError("Cannot compare Solves with x0 set")

    def __variable_attrs__(self) -> Tuple[str]:
        return 'x0',


class SolveResult(Generic[X, Y]):
    """
    Stores information about the solution or trajectory of a solve.

    When representing the full optimization trajectory, all tracked quantities will have an additional `trajectory` batch dimension.
    """

    def __init__(self,
                 solve: Solve,
                 x: X,
                 residual: Y or None,
                 iterations: Tensor or None,
                 function_evaluations: Tensor or None,
                 converged: Tensor,
                 diverged: Tensor,
                 method: str,
                 msg: str = None):
        # tuple.__new__(SolveResult, (x, residual, iterations, function_evaluations, converged, diverged))
        self.solve: Solve[X, Y] = solve
        """ `Solve`, Parameters specified for the solve. """
        self.x: X = x
        """ `Tensor` or `TensorLike`, solution estimate. """
        self.residual: Y = residual
        """ `Tensor` or `TensorLike`, residual vector for systems of equations or function value for minimization problems. """
        self.iterations: Tensor = iterations
        """ `Tensor`, number of performed iterations to reach this state. """
        self.function_evaluations: Tensor = function_evaluations
        """ `Tensor`, how often the function (or its gradient function) was called. """
        self.converged: Tensor = converged
        """ `Tensor`, whether the residual is within the specified tolerance. """
        self.diverged: Tensor = diverged
        """ `Tensor`, whether the solve has diverged at this point. """
        self.method = method
        """ `str`, which method and implementation that was used. """
        if not msg:
            if self.diverged.any:
                msg = f"Solve diverged within {iterations if iterations is not None else '?'} iterations using {method}."
            elif not self.converged.trajectory[-1].all:
                msg = f"Solve did not converge to rel={solve.relative_tolerance}, abs={solve.absolute_tolerance} within {solve.max_iterations} iterations using {method}."
            else:
                msg = f"Converged within {iterations if iterations is not None else '?'} iterations."
        self.msg = msg
        """ `str`, termination message """

    def __repr__(self):
        return self.msg

    def snapshot(self, index):
        return SolveResult(self.solve, self.x.trajectory[index], self.residual.trajectory[index], self.iterations.trajectory[index], self.function_evaluations.trajectory[index], self.converged.trajectory[index], self.diverged.trajectory[index], self.method, self.msg)

    def convergence_check(self, only_warn):
        if self.diverged.any:
            if Diverged not in self.solve.suppress:
                if only_warn:
                    warnings.warn(self.msg)
                else:
                    raise Diverged(self)
        if not self.converged.trajectory[-1].all:
            if NotConverged not in self.solve.suppress:
                if only_warn:
                    warnings.warn(self.msg)
                else:
                    raise NotConverged(self)


class ConvergenceException(RuntimeError):
    """
    Base class for exceptions raised when a solve does not converge.

    See Also:
        `Diverged`, `NotConverged`.
    """

    def __init__(self, result: SolveResult):
        RuntimeError.__init__(self, result)
        self.result: SolveResult = result
        """ `SolveResult` holding information about the solve. """


class NotConverged(ConvergenceException):
    """
    Raised during optimization if the desired accuracy was not reached within the maximum number of iterations.

    This exception inherits from `ConvergenceException`.

    See Also:
        `Diverged`.
    """

    def __init__(self, result: SolveResult):
        ConvergenceException.__init__(self, result)


class Diverged(ConvergenceException):
    """
    Raised if the optimization was stopped prematurely and cannot continue.
    This may indicate that no solution exists.

    The values of the last estimate `x` may or may not be finite.

    This exception inherits from `ConvergenceException`.

    See Also:
        `NotConverged`.
    """

    def __init__(self, result: SolveResult):
        ConvergenceException.__init__(self, result)


class SolveTape:

    def __init__(self, record_trajectories=False):
        """
        Used to record additional information about solves invoked via `solve_linear()`, `solve_nonlinear()` or `minimize()`.

        To access a `SolveResult` of a recorded solve, use
        ```python
        solve = Solve(method, ...)
        with SolveTape() as solves:
            x = math.solve_linear(f, y, solve)
        result: SolveResult = solves[solve]  # get by Solve
        result: SolveResult = solves[0]  # get by index
        ```

        Args:
            record_trajectories: When enabled, the entries of `SolveResult` will contain an additional batch dimension named `trajectory`.
        """
        self.record_trajectories = record_trajectories
        self.solves: List[SolveResult] = []
        self.solve_ids: List[str] = []

    def __enter__(self):
        _SOLVE_TAPES.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _SOLVE_TAPES.remove(self)

    def _add(self, solve: Solve, mode: type, result: SolveResult):
        assert all(s.solve.id != solve.id for s in self.solves)
        if self.record_trajectories:
            assert mode == list
            self.solves.append(result)
        elif mode == list:
            self.solves.append(result.snapshot(-1))
        else:
            self.solves.append(result)
        self.solve_ids.append(solve.id)

    def __getitem__(self, item) -> SolveResult:
        if isinstance(item, int):
            return self.solves[item]
        else:
            assert isinstance(item, Solve)
            solves = [s for s in self.solves if s.solve.id == item.id]
            if len(solves) == 0:
                raise KeyError(f"No solve recorded with key '{item}'.")
            assert len(solves) == 1
            return solves[0]

    def __len__(self):
        return len(self.solves)


_SOLVE_TAPES: List[SolveTape] = []


def current_solve_mode():
    if not _SOLVE_TAPES:
        return BasicSolveResult
    elif any(t.record_trajectories for t in _SOLVE_TAPES):
        return list
    else:
        return FullSolveResult


def minimize(f: Callable[[X], Y], solve: Solve[X, Y]) -> X:
    """
    Finds a minimum of the scalar function *f(x)*.
    The `method` argument of `solve` determines which method is used.
    All methods supported by `scipy.optimize.minimize` are supported,
    see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html .

    This method is limited to backends that support `functional_gradient()`, currently PyTorch, TensorFlow and Jax.

    To obtain additional information about the performed solve, use a `SolveTape`.

    See Also:
        `solve_nonlinear()`.

    Args:
        f: Function whose output is subject to minimization.
            All positional arguments of `f` are optimized and must be `Tensor` or `TensorLike`.
            The first return value of `f` must be a scalar float `Tensor` or `TensorLike`.
        solve: `Solve` object to specify method type, parameters and initial guess for `x`.

    Returns:
        x: solution, the minimum point `x`.

    Raises:
        NotConverged: If the desired accuracy was not be reached within the maximum number of iterations.
        Diverged: If the optimization failed prematurely.
    """
    assert solve.relative_tolerance == 0, f"relative_tolerance must be zero for minimize() but got {solve.relative_tolerance}"
    x0_nest, x0_tensors = disassemble_nested(solve.x0)
    backend = choose_backend_t(*x0_tensors, prefer_default=True)
    batch = combine_safe(*[t.shape for t in x0_tensors]).batch
    x0_natives = []
    for t in x0_tensors:
        t._expand()
        assert t.shape.is_uniform
        x0_natives.append(reshaped_native(t, [batch, t.shape.non_batch], force_expand=True))
    x0_flat = backend.concat(x0_natives, -1)

    def unflatten_assemble(x_flat):
        i = 0
        x_tensors = []
        for x0_native, x0_tensor in zip(x0_natives, x0_tensors):
            vol = backend.shape(x0_native)[-1]
            flat_native = x_flat[:, i:i + vol]
            x_tensors.append(reshaped_tensor(flat_native, [batch, x0_tensor.shape.non_batch]))
            i += vol
        x = assemble_nested(x0_nest, x_tensors)
        return x

    def native_function(x_flat):
        x = unflatten_assemble(x_flat)
        if isinstance(x, (tuple, list)):
            y = f(*x)
        else:
            y = f(x)
        _, y_tensors = disassemble_nested(y)
        return y_tensors[0].native()

    def assemble_state(native: SolverState):
        x = unflatten_assemble(native.x)
        return SolverState(x, native.residual, native.iteration, native.function_evaluations, native.converged, native.diverged)

    def assemble_result(native: SolveResult):
        result = SolveResult(solve)
        result.final_state = assemble_state(native.final_state)
        return result

    result_native = backend.minimize(native_function, copy_with(solve, x0=x0_flat))
    result = assemble_result(result_native)
    result.convergence_check()
    return result


def solve_nonlinear(f: Callable, y, solve: Solve) -> Tensor:
    """
    Solves the non-linear equation *f(x) = y* by minimizing the norm of the residual.

    This method is limited to backends that support `functional_gradient()`, currently PyTorch, TensorFlow and Jax.

    To obtain additional information about the performed solve, use a `SolveTape`.

    See Also:
        `minimize()`, `solve_linear()`.

    Args:
        f: Function whose output is optimized to match `y`.
            All positional arguments of `f` are optimized and must be `Tensor` or `TensorLike`.
            The output of `f` must match `y`.
        y: Desired output of `f(x)` as `Tensor` or `TensorLike`.
        solve: `Solve` object specifying optimization method, parameters and initial guess for `x`.

    Returns:
        x: Solution fulfilling `f(x) = y` within specified tolerance as `Tensor` or `TensorLike`.

    Raises:
        NotConverged: If the desired accuracy was not be reached within the maximum number of iterations.
        Diverged: If the solve failed prematurely.
    """
    from ._nd import l2_loss

    def min_func(x):
        diff = f(x) - y
        l2 = l2_loss(diff)
        return l2

    rel_tol_to_abs = solve.relative_tolerance * l2_loss(y, batch_norm=True)
    solve.absolute_tolerance = rel_tol_to_abs
    solve.relative_tolerance = 0
    return minimize(min_func, solve)


def solve_linear(f: Callable[[X], Y], y: Y, solve: Solve[X, Y]) -> X:
    """
    Solves the system of linear equations *f(x) = y* and returns *x*.
    For maximum performance, compile `f` using `jit_compile_linear()` beforehand.
    This will use a matrix representation of `f` to solve the linear system.

    To obtain additional information about the performed solve, use a `SolveTape`.

    The gradient of this operation will perform another linear solve with the parameters specified by `Solve.gradient_solve`.

    See Also:
        `solve_nonlinear()`, `jit_compile_linear()`.

    Args:
        f: Linear function with single `Tensor` or `TensorLike` positional argument and return value.
        y: Desired output of `f(x)` as `Tensor` or `TensorLike`.
        solve: `Solve` object specifying optimization method, parameters and initial guess for `x`.

    Returns:
        x: solution of the linear system of equations `f(x) = y` as `Tensor` or `TensorLike`.

    Raises:
        NotConverged: If the desired accuracy was not be reached within the maximum number of iterations.
        Diverged: If the solve failed prematurely.
    """
    y_nest, y_tensors = disassemble_nested(y)
    x0_nest, x0_tensors = disassemble_nested(solve.x0)
    assert len(x0_tensors) == len(y_tensors) == 1, "Only single-tensor linear solves are currently supported"
    backend = choose_backend_t(*y_tensors, *x0_tensors)

    if not all_available(*y_tensors, *x0_tensors):  # jit mode
        f = jit_compile_linear(f) if backend.supports(Backend.sparse_tensor) else jit_compile(f)

    if isinstance(f, LinearFunction) and backend.supports(Backend.sparse_tensor):
        matrix = f.sparse_coordinate_matrix(solve.x0)
        return _matrix_solve(y, solve, matrix, backend=backend)  # custom_gradient
    else:
        return _function_solve(y, solve, f=f, backend=backend)  # custom_gradient


def _linear_solve_forward(y, solve: Solve, native_lin_op,
                          active_dims: Shape or None, backend: Backend, is_backprop: bool) -> Any:
    y_nest, (y_tensor,) = disassemble_nested(y)
    x0_nest, (x0_tensor,) = disassemble_nested(solve.x0)
    batch = (y_tensor.shape & x0_tensor.shape).without(active_dims)
    x0_native = backend.as_tensor(reshaped_native(x0_tensor, [batch, active_dims], force_expand=True))
    y_native = backend.as_tensor(reshaped_native(y_tensor, [batch, active_dims], force_expand=True))
    rtol = backend.to_float(reshaped_native(solve.relative_tolerance, [batch], force_expand=True))
    atol = backend.to_float(reshaped_native(solve.absolute_tolerance, [batch], force_expand=True))
    maxi = backend.to_int32(reshaped_native(solve.max_iterations, [batch], force_expand=True))
    ret_type = current_solve_mode()
    ret = backend.linear_solve(solve.method, native_lin_op, y_native, x0_native, rtol, atol, maxi, ret_type)
    if isinstance(ret, BasicSolveResult):
        converged = reshaped_tensor(ret.converged, [batch])
        diverged = reshaped_tensor(ret.diverged, [batch])
        x = assemble_nested(x0_nest, [reshaped_tensor(ret.x, [batch, active_dims])])
        msg = "Re-run solve with a SolveTape to get additional information."
        result = SolveResult(solve, x, None, None, None, converged, diverged, ret.method, msg)
    elif isinstance(ret, FullSolveResult):
        converged = reshaped_tensor(ret.converged, [batch])
        diverged = reshaped_tensor(ret.diverged, [batch])
        x = assemble_nested(x0_nest, [reshaped_tensor(ret.x, [batch, active_dims])])
        iterations = reshaped_tensor(ret.iterations, [batch])
        function_evaluations = reshaped_tensor(ret.function_evaluations, [batch])
        residual = assemble_nested(y_nest, [reshaped_tensor(ret.residual, [batch, active_dims])])
        result = SolveResult(solve, x, residual, iterations, function_evaluations, converged, diverged, ret.method, ret.message)
    elif isinstance(ret, (tuple, list)):  # trajectory
        assert all(isinstance(r, FullSolveResult) for r in ret)
        converged = batch_stack([reshaped_tensor(r.converged, [batch]) for r in ret], 'trajectory')
        diverged = batch_stack([reshaped_tensor(r.diverged, [batch]) for r in ret], 'trajectory')
        x = assemble_nested(x0_nest, [reshaped_tensor(ret[-1].x, [batch, active_dims])])
        x_ = assemble_nested(x0_nest, [batch_stack([reshaped_tensor(r.x, [batch, active_dims]) for r in ret], 'trajectory')])
        residual = assemble_nested(y_nest, [batch_stack([reshaped_tensor(r.residual, [batch, active_dims]) for r in ret], 'trajectory')])
        iterations = reshaped_tensor(ret[-1].iterations, [batch])
        function_evaluations = batch_stack([reshaped_tensor(r.function_evaluations, [batch]) for r in ret], 'trajectory')
        result = SolveResult(solve, x_, residual, iterations, function_evaluations, converged, diverged, ret[-1].method, ret[-1].message)
    else:
        raise AssertionError(f"Backend.linear_solve returned invalid result: {type(ret)}")
    for tape in _SOLVE_TAPES:
        tape._add(solve, ret_type, result)
    result.convergence_check(is_backprop and backend.name == 'TensorFlow')  # raises ConvergenceException
    return x


def attach_gradient_solve(forward_solve: Callable):
    def implicit_gradient_solve(*args, **kwargs):
        y, solve, *matrix, x, dx = args
        grad_solve = solve.gradient_solve
        x0 = grad_solve.x0 if grad_solve.x0 is not None else zeros_like(solve.x0)
        grad_solve_ = copy_with(solve.gradient_solve, x0=x0)
        dy = solve_with_grad(dx, grad_solve_, *matrix, is_backprop=True, **kwargs)
        return (dy, None, *([None] * len(matrix)))  # this should hopefully result in implicit gradients for higher orders as well
    solve_with_grad = custom_gradient(forward_solve, implicit_gradient_solve)
    return solve_with_grad


def _matrix_solve_forward(y, solve: Solve, matrix: FixedShiftSparseTensor,
                          backend: Backend = None, is_backprop=False):  # kwargs
    matrix_native = matrix.native()
    active_dims = matrix.src_shape
    result = _linear_solve_forward(y, solve, matrix_native, active_dims=active_dims, backend=backend, is_backprop=is_backprop)
    return result  # must return exactly `x` so gradient isn't computed w.r.t. other quantities


_matrix_solve = attach_gradient_solve(_matrix_solve_forward)


def _function_solve_forward(y, solve: Solve,
                            f: Callable = None, backend: Backend = None, is_backprop=False):  # kwargs
    y_nest, (y_tensor,) = disassemble_nested(y)
    x0_nest, (x0_tensor,) = disassemble_nested(solve.x0)
    active_dims = (y_tensor.shape & x0_tensor.shape).non_batch  # assumes batch dimensions are not active
    batch = (y_tensor.shape & x0_tensor.shape).batch

    def native_f(native_x):
        x = assemble_nested(x0_nest, [reshaped_tensor(native_x, [batch, active_dims] if backend.ndims(native_x) >= 2 else [active_dims])])
        y = f(x)
        _, (y_tensor,) = disassemble_nested(y)
        y_native = reshaped_native(y_tensor, [batch, active_dims] if backend.ndims(native_x) >= 2 else [active_dims])
        return y_native

    result = _linear_solve_forward(y, solve, native_f, active_dims=active_dims, backend=backend, is_backprop=is_backprop)
    return result  # must return exactly `x` so gradient isn't computed w.r.t. other quantities


_function_solve = attach_gradient_solve(_function_solve_forward)


def print_gradient(value: Tensor, name="", detailed=False) -> Tensor:
    """
    Prints the gradient vector of `value` when computed.
    The gradient at `value` is the vector-Jacobian product of all operations between the output of this function and the loss value.

    The gradient is not printed in jit mode, see `jit_compile()`.

    Example:
        ```python
        def f(x):
            x = math.print_gradient(x, 'dx')
            return math.l1_loss(x)

        math.functional_gradient(f)(math.ones(x=6))
        ```

    Args:
        value: `Tensor` for which the gradient may be computed later.
        name: (Optional) Name to print along with the gradient values
        detailed: If `False`, prints a short summary of the gradient tensor.

    Returns:
        `identity(value)` which when differentiated, prints the gradient vector.
    """
    def print_grad(_x, _y, dx):
        if all_available(_x, dx):
            if detailed:
                print_(dx, name=name)
            else:
                print(f"{name}:  \t{dx}")
        return dx,
    identity = custom_gradient(lambda x: x, print_grad)
    return identity(value)

