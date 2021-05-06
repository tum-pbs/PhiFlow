import time
import warnings
from functools import reduce
from typing import Tuple, Callable, Dict

import numpy as np

from . import _ops as math
from ._shape import EMPTY_SHAPE, Shape, parse_dim_order, SPATIAL_DIM
from ._tensors import Tensor, NativeTensor, CollapsedTensor
from .backend import choose_backend, Backend, get_current_profile


def disassemble(obj: Tensor or tuple or list, expand=True):
    assert isinstance(obj, (Tensor, tuple, list)), f"jit-compiled function returned {type(obj)} but must return either a 'phi.math.Tensor' or tuple/list of tensors."
    if isinstance(obj, Tensor):
        if expand:
            obj._expand()
        return obj._natives(), obj.shape
    else:
        for t in obj:
            t._expand()
        return sum([t._natives() for t in obj], ()), tuple(t.shape for t in obj)


def assemble(natives: tuple, shapes: Shape or Tuple[Shape]):
    natives = list(natives)
    if isinstance(shapes, Shape):
        return _assemble_pop(natives, shapes)
    else:
        return [_assemble_pop(natives, shape) for shape in shapes]


def _assemble_pop(natives: list, shape: Shape):
    if shape.is_uniform:
        native = natives.pop(0)
        return NativeTensor(native, shape)
    else:
        s2 = shape.shape.without('dims')
        if len(s2) > 1:
            raise NotImplementedError('More than one non-uniform dimension not supported.')
        shapes = shape.unstack(s2.name)
        tensors = [NativeTensor(natives.pop(0), s) for s in shapes]
        return math._stack(tensors, s2.name, s2.types[0])


def match_output_shapes(input_shapes: Tuple[Shape], transforms: Dict[Tuple[Shape], Shape or Tuple[Shape]]):
    # --- Search for a perfect shape match ---
    for prev_input_shapes, prev_output in transforms.items():
        assert len(prev_input_shapes) == len(input_shapes)
        if all(prev_shape == i_shape for prev_shape, i_shape in zip(prev_input_shapes, input_shapes)):
            return prev_output
    # --- Search for a names match ---
    for prev_input_shapes, prev_output in transforms.items():
        if all(prev_shape.names == i_shape.names for prev_shape, i_shape in zip(prev_input_shapes, input_shapes)):
            if isinstance(prev_output, Shape):
                return extrapolate_shape(prev_output, prev_input_shapes, input_shapes)
            else:
                return tuple(extrapolate_shape(o, prev_input_shapes, input_shapes) for o in prev_output)
    raise KeyError(f"Not output shape found for input shapes {input_shapes}."
                   f" Maybe the backend extrapolated the concrete function from another trace?"
                   f" Registered transforms: {transforms}")


def extrapolate_shape(prev_out: Shape, prev_in: Tuple[Shape], new_in: Tuple[Shape]):
    sizes = []
    for dim, size in prev_out.named_sizes:
        for p_in, n_in in zip(prev_in, new_in):
            if dim in p_in and size == p_in.get_size(dim):
                sizes.append(n_in.get_size(dim))
                break
        else:
            raise ValueError(prev_out, prev_in, new_in)
    return prev_out.with_sizes(sizes)


class JitFunction:

    def __init__(self, f: Callable):
        self.f = f
        self.traces: Dict[Backend, Callable] = {}
        self.shape_transform: Dict[tuple, tuple or Shape] = {}
        self._current_input_shapes: tuple = ()

    def _jit_compile(self, backend: Backend):
        def f_native(*natives, **kwargs):
            values = assemble(natives, self._current_input_shapes)
            result = self.f(*values, **kwargs)  # Tensor or tuple/list of Tensors
            result_native, self.shape_transform[self._current_input_shapes] = disassemble(result)
            return result_native

        self.traces[backend] = backend.jit_compile(f_native)

    def __call__(self, *args, **kwargs):
        assert not kwargs
        backend = math.choose_backend_t(*args)
        if not backend.supports(Backend.jit_compile):
            warnings.warn(f"jit_copmile() not supported by {backend}. Running function '{self.f.__name__}' as-is.")
            return self.f(*args)
        natives, self._current_input_shapes = disassemble(args)
        if backend not in self.traces:
            self._jit_compile(backend)
        native_result = self.traces[backend](*natives)
        result_shapes = match_output_shapes(self._current_input_shapes, self.shape_transform)
        return assemble(native_result, result_shapes)

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
        Function with similar signature and return values as `f`. However, the returned function does not support keyword arguments.
    """
    return f if isinstance(f, (JitFunction, LinearFunction)) else JitFunction(f)


class LinearFunction:
    """
    Just-in-time compiled linear function of `Tensor` arguments and return values.

    Use `jit_compile_linear()` to create a linear function representation.
    """

    def __init__(self, f):
        self.f = f
        self.tracers = {}  # Shape -> Backend -> ShiftLinTracer
        self.nl_jit = JitFunction(f)  # for backends that do not support sparse matrices

    def __call__(self, *args, **kwargs):
        assert not kwargs, "kwargs not supported, pass all values as positional arguments."
        if len(args) > 1:
            return self.nl_jit(*args)
        arg = args[0]
        if isinstance(arg, ShiftLinTracer):
            # TODO if already compiled, use cached ShiftLinTracer
            return self.f(arg)
        backend = math.choose_backend_t(arg)
        if not backend.supports(Backend.sparse_tensor):
            warnings.warn(f"Sparse matrices are not supported by {backend}. Falling back to regular jit compilation.")
            return self.nl_jit(*args)
        tracer = self._tracer(arg)
        return tracer.apply(arg)

    def _tracer(self, x):
        backend = math.choose_backend_t(x)
        if x.shape in self.tracers:
            if backend in self.tracers[x.shape]:
                return self.tracers[x.shape][backend]
        return self._trace(x)

    def sparse_coordinate_matrix(self, *args: Tensor, **kwargs):
        assert not kwargs, "kwargs not supported, pass all values as positional arguments."
        assert all(isinstance(arg, Tensor) for arg in args)
        if len(args) > 1:
            raise NotImplementedError("Matrix can currently only be built for functions with a single input.")
        arg = args[0]
        backend = math.choose_backend_t(arg)
        assert backend.supports(Backend.sparse_tensor)
        if arg.shape in self.tracers and backend in self.tracers[arg.shape]:
            return self.tracers[arg.shape][backend].get_sparse_coordinate_matrix()
        return self._trace(arg).get_sparse_coordinate_matrix()

    def _trace(self, x: Tensor) -> 'ShiftLinTracer':
        trace_time = time.perf_counter()
        x = math.to_float(x)
        with x.default_backend:
            tracer = ShiftLinTracer(x, {EMPTY_SHAPE: math.ones(dtype=x.dtype)}, x.shape)
            result = self.f(tracer)
        assert isinstance(result, ShiftLinTracer), f"Tracing linear function '{self.f.__name__}' failed. Make sure only linear operations are used."
        trace_time = time.perf_counter() - trace_time
        if get_current_profile():
            get_current_profile().add_external_message(f"Linear trace of '{self.f.__name__}': {round(trace_time * 1000)} ms.")
        if x.shape not in self.tracers:
            self.tracers[x.shape] = {}
        self.tracers[x.shape][math.choose_backend_t(x)] = result
        return result

    def dense_stencil(self, x: Tensor, multi_index: Tensor = None, **indices):
        if multi_index is None:
            multi_index = shape(indices).sort(x.shape)
            multi_index = math.tensor(multi_index, 'vector')
        tracer = self._tracer(x)



def jit_compile_linear(f: Callable) -> 'LinearFunction':
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
        `LinearFunction` with similar signature and return values as `f`. However, the returned function does not support keyword arguments.
    """
    return f if isinstance(f, LinearFunction) else LinearFunction(f)


def is_tracer(t: Tensor):
    return isinstance(t, ShiftLinTracer)


def simplify_add(val: dict):
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
        self.val = simplify_add(values_by_shift)
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
        mat = self.get_sparse_coordinate_matrix()
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

    def get_sparse_coordinate_matrix(self):
        """
        Builds a sparse matrix that represents this linear operation.
        Independent dimensions, those that can be treated as batch dimensions, are recognized automatically and ignored.
        
        :return: native sparse tensor

        Args:

        Returns:

        """
        if self._sparse_coo is not None:
            return self._sparse_coo
        build_time = time.perf_counter()
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
        # TODO sort indices
        self._sparse_coo = choose_backend(rows, cols, vals).sparse_tensor((rows, cols), vals, (out_shape.volume, src_shape.volume))
        build_time = time.perf_counter() - build_time
        if get_current_profile():
            get_current_profile().add_external_message(f"build: {round(build_time * 1000)} ms")
        return self._sparse_coo

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
