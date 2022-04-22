import time
import types
import uuid
import warnings
from functools import wraps
from typing import Tuple, Callable, Dict, Generic, List, TypeVar, Any

import numpy as np

from . import _ops as math
from ._ops import choose_backend_t, zeros_like, all_available, print_, reshaped_native, reshaped_tensor, stack, to_float
from ._shape import EMPTY_SHAPE, Shape, parse_dim_order, vector_add, merge_shapes, spatial, instance, batch, concat_shapes
from ._tensors import Tensor, NativeTensor, disassemble_tree, TensorLike, assemble_tree, copy_with, disassemble_tensors, assemble_tensors, variable_attributes, wrap, cached
from .backend import choose_backend, Backend
from .backend._backend import SolveResult, get_spatial_derivative_order, functional_derivative_evaluation, PHI_LOGGER

X = TypeVar('X')
Y = TypeVar('Y')


class SignatureKey:

    def __init__(self,
                 source_function: Callable or None,
                 tree,
                 shapes: Shape or Tuple[Shape],
                 kwargs: dict or None,
                 backend: Backend,
                 tracing: bool):
        assert isinstance(tree, TensorLike), tree
        if source_function is None:  # this is an input signature
            assert isinstance(shapes, tuple)
        self.source_function = source_function
        self.tree = tree
        self.shapes = shapes
        self.kwargs = kwargs
        self.backend = backend
        self.tracing = tracing
        self.spatial_derivative_order = get_spatial_derivative_order()

    def __repr__(self):
        return f"{self.tree} with shapes {self.shapes}"

    def __eq__(self, other: 'SignatureKey'):
        assert isinstance(other, SignatureKey)
        return self.tree == other.tree and self.shapes == other.shapes and self.kwargs == other.kwargs and self.backend == other.backend and self.spatial_derivative_order == other.spatial_derivative_order

    def __hash__(self):
        return hash(self.shapes) + hash(self.backend)

    def matches_structure_and_names(self, other: 'SignatureKey'):
        assert isinstance(other, SignatureKey)
        return self.tree == other.tree and all(s1.names == s2.names for s1, s2 in zip(self.shapes, other.shapes)) and self.kwargs == other.kwargs and self.backend == other.backend

    def extrapolate(self, rec_in: 'SignatureKey', new_in: 'SignatureKey') -> 'SignatureKey':
        assert self.source_function is not None, "extrapolate() must be called on output keys"
        shapes = [self._extrapolate_shape(s, rec_in, new_in) for s in self.shapes]
        return SignatureKey(self.source_function, self.tree, shapes, self.kwargs, self.backend)

    @staticmethod
    def _extrapolate_shape(shape_: Shape, rec_in: 'SignatureKey', new_in: 'SignatureKey') -> Shape:
        sizes = []
        for dim, size in shape_._named_sizes:
            for p_in, n_in in zip(rec_in.shapes, new_in.shapes):
                if dim in p_in and size == p_in.get_size(dim):
                    sizes.append(n_in.get_size(dim))
                    break
            else:
                raise ValueError(shape_, rec_in, new_in)
        return shape_.with_sizes(sizes)


def match_output_signature(new_in: SignatureKey, recorded_mappings: Dict[SignatureKey, SignatureKey], source) -> SignatureKey:
    for rec_in, rec_out in recorded_mappings.items():
        if rec_in == new_in:  # exact match
            return rec_out
    for rec_in, rec_out in recorded_mappings.items():
        if rec_in.matches_structure_and_names(new_in):
            return rec_out.extrapolate(rec_in, new_in)
    transforms_str = ''.join([f'\n* {i} -> {o}' for i, o in recorded_mappings.items()])
    raise RuntimeError(f"{source}: no output shape found for input shapes {new_in}.\n"
                       f"Maybe the backend extrapolated the concrete function from another trace?\n"
                       f"Registered transforms:\n{transforms_str}")  # KeyError does not support \n


def key_from_args(*args, cache=False, **kwargs) -> Tuple[SignatureKey, list]:
    nest, tensors = disassemble_tree(args)
    tracing = not all_available(*tensors)
    backend = math.choose_backend_t(*tensors)
    # if tracing and cache:
    #     cache = False
    #     warnings.warn("Cannot cache a tensor while tracing.", RuntimeWarning)
    natives, shapes = disassemble_tensors(tensors, expand=cache)
    key = SignatureKey(None, nest, shapes, kwargs, backend, tracing)
    return key, natives


def key_from_args_pack_batch(*args, cache=False, **kwargs) -> Tuple[SignatureKey, list, Shape]:
    nest, tensors = disassemble_tree(args)
    tracing = not all_available(*tensors)
    backend = math.choose_backend_t(*tensors)
    # if tracing and cache:
    #     cache = False
    #     warnings.warn("Cannot cache a tensor while tracing.", RuntimeWarning)
    batch_shape = merge_shapes(*[t.shape.batch for t in tensors])
    # tensors = [math.pack_dims(t, batch_shape, batch('batch'), pos=0) for t in tensors]
    natives = [math.reshaped_native(t, [batch_shape, *t.shape.non_batch], force_expand=True) for t in tensors]
    # natives, shapes = disassemble_tensors(tensors, expand=cache)
    shapes = tuple([math.concat_shapes(batch(batch=batch_shape.volume), *t.shape.non_batch) for t in tensors])
    key = SignatureKey(None, nest, shapes, kwargs, backend, tracing)
    return key, natives, batch_shape


class JitFunction:

    def __init__(self, f: Callable):
        self.f = f
        self.traces: Dict[SignatureKey, Callable] = {}
        self.recorded_mappings: Dict[SignatureKey, SignatureKey] = {}
        self.grad_jit = GradientFunction(f.f, f.wrt, f.get_output, jit=True) if isinstance(f, GradientFunction) else None

    def _jit_compile(self, in_key: SignatureKey):
        PHI_LOGGER.debug(f"Φ-jit: '{self.f.__name__}' called with new key. shapes={[s.volume for s in in_key.shapes]}, kwargs={list(in_key.kwargs)}.")

        def jit_f_native(*natives, **kwargs):
            PHI_LOGGER.debug(f"Φ-jit: Tracing '{self.f.__name__}'")
            assert not kwargs
            in_tensors = assemble_tensors(natives, in_key.shapes)
            values = assemble_tree(in_key.tree, in_tensors)
            assert isinstance(values, tuple)  # was disassembled from *args
            result = self.f(*values, **in_key.kwargs)  # Tensor or tuple/list of Tensors
            nest, out_tensors = disassemble_tree(result)
            result_natives, result_shapes = disassemble_tensors(out_tensors)
            self.recorded_mappings[in_key] = SignatureKey(jit_f_native, nest, result_shapes, None, in_key.backend, in_key.tracing)
            return result_natives

        jit_f_native.__name__ = f"native({self.f.__name__ if isinstance(self.f, types.FunctionType) else str(self.f)})"
        return in_key.backend.jit_compile(jit_f_native)

    def __call__(self, *args, **kwargs):
        key, natives = key_from_args(*args, cache=True, **kwargs)
        if isinstance(self.f, GradientFunction) and key.backend.supports(Backend.jit_compile_grad):
            return self.grad_jit(*args, **kwargs)
        if not key.backend.supports(Backend.jit_compile):
            warnings.warn(f"jit_copmile() not supported by {key.backend}. Running function '{self.f.__name__}' as-is.", RuntimeWarning)
            return self.f(*args, **kwargs)
        if key not in self.traces:
            self.traces[key] = self._jit_compile(key)
        native_result = self.traces[key](*natives)
        output_key = match_output_signature(key, self.recorded_mappings, self)
        output_tensors = assemble_tensors(native_result, output_key.shapes)
        return assemble_tree(output_key.tree, output_tensors)

    def __repr__(self):
        return f"jit({self.f.__name__})"

    @property
    def __name__(self):
        return self.f.__name__


def jit_compile(f: Callable) -> Callable:
    """
    Compiles a graph based on the function `f`.
    The graph compilation is performed just-in-time (jit), e.g. when the returned function is called for the first time.

    The traced function will compute the same result as `f` but may run much faster.
    Some checks may be disabled in the compiled function.

    Can be used as a decorator:
    ```python
    @math.jit_compile
    def my_function(x: math.Tensor) -> math.Tensor:
    ```

    Invoking the returned function may invoke re-tracing / re-compiling `f` after the first call if either

    * it is called with a different number of arguments,
    * the keyword arguments differ from previous invocations,
    * the positional tensor arguments have different dimension names or types (the dimension order also counts),
    * any positional `Tensor` arguments require a different backend than previous invocations,
    * `TensorLike` positional arguments do not match in non-variable properties.

    Compilation is implemented for the following backends:

    * PyTorch: [`torch.jit.trace`](https://pytorch.org/docs/stable/jit.html)
    * TensorFlow: [`tf.function`](https://www.tensorflow.org/guide/function)
    * Jax: [`jax.jit`](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html#using-jit-to-speed-up-functions)

    Jit-compilations cannot be nested, i.e. you cannot call `jit_compile()` while another function is being compiled.
    An exception to this is `jit_compile_linear()` which can be called from within a jit-compiled function.

    See Also:
        `jit_compile_linear()`

    Args:
        f: Function to be traced.
            All positional arguments must be of type `Tensor` or `TensorLike` returning a single `Tensor` or `TensorLike`.

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
        assert in_key.shapes[0].is_uniform, f"math.jit_compile_linear() only supports uniform tensors for function input and output but input shape was {in_key.shapes[0]}"
        with in_key.backend:
            x = math.ones(in_key.shapes[0])
            tracer = ShiftLinTracer(x, {EMPTY_SHAPE: math.ones()}, x.shape, math.zeros(x.shape))
        f_input = assemble_tree(in_key.tree, [tracer])
        assert isinstance(f_input, tuple)
        condition_args = [in_key.kwargs[f'_condition_arg[{i}]'] for i in range(in_key.kwargs['n_condition_args'])]
        kwargs = {k: v for k, v in in_key.kwargs.items() if not (k.startswith('_condition_arg[') or k == 'n_condition_args')}
        result = self.f(*f_input, *condition_args, **kwargs)
        _, result_tensors = disassemble_tree(result)
        assert len(result_tensors) == 1, f"Linear function must return a single Tensor or tensor-like but got {result}"
        result_tensor = result_tensors[0]
        assert isinstance(result_tensor, ShiftLinTracer), f"Tracing linear function '{self.f.__name__}' failed. Make sure only linear operations are used."
        return result_tensor

    def _get_or_trace(self, key: SignatureKey):
        if not key.tracing and key in self.tracers:
            return self.tracers[key]
        else:
            tracer = self._trace(key)
            if not key.tracing:
                self.tracers[key] = tracer
                if len(self.tracers) >= 4:
                    warnings.warn(f"Φ-lin: The compiled linear function '{self.f.__name__}' was traced {len(self.tracers)} times. Performing many traces may be slow and cause memory leaks. A trace is performed when the function is called with different keyword arguments. Multiple linear traces can be avoided by jit-compiling the code that calls jit_compile_linear().", RuntimeWarning)
            return tracer

    def __call__(self, *args: X, **kwargs) -> Y:
        nest, tensors = disassemble_tree(args)
        assert tensors, "Linear function requires at least one argument"
        if any(isinstance(t, ShiftLinTracer) for t in tensors):
            # TODO: if t is identity, use cached ShiftLinTracer, otherwise multiply two ShiftLinTracers
            return self.f(*args, **kwargs)
        backend = math.choose_backend_t(*tensors)
        if not backend.supports(Backend.sparse_coo_tensor):
            # warnings.warn(f"Sparse matrices are not supported by {backend}. Falling back to regular jit compilation.", RuntimeWarning)
            if not all_available(*tensors):  # avoid nested tracing, Typical case jax.scipy.sparse.cg(LinearFunction). Nested traces cannot be reused which results in lots of traces per cg.
                PHI_LOGGER.debug(f"Φ-lin: Running '{self.f.__name__}' as-is with {backend} because it is being traced.")
                return self.f(*args, **kwargs)
            else:
                return self.nl_jit(*args, **kwargs)
        x, *condition_args = args
        key = self._condition_key(x, condition_args, kwargs)
        tracer = self._get_or_trace(key)
        return tracer.apply(tensors[0])

    def sparse_matrix(self, x, *condition_args, format: str = None, **kwargs):
        key = self._condition_key(x, condition_args, kwargs)
        tracer = self._get_or_trace(key)
        assert math.close(tracer.bias, 0), "This is an affine function and cannot be represented by a single matrix. Use sparse_matrix_and_bias() instead."
        return tracer.get_sparse_matrix(format)

    def sparse_matrix_and_bias(self, x, *condition_args, format: str = None, **kwargs):
        key = self._condition_key(x, condition_args, kwargs)
        tracer = self._get_or_trace(key)
        return tracer.get_sparse_matrix(format), tracer.bias

    def _condition_key(self, x, condition_args, kwargs):
        kwargs['n_condition_args'] = len(condition_args)
        for i, c_arg in enumerate(condition_args):
            kwargs[f'_condition_arg[{i}]'] = c_arg
        key, _ = key_from_args(x, cache=False, **kwargs)
        # assert key.backend.supports(Backend.sparse_coo_tensor)
        return key

    def stencil_inspector(self, *args, **kwargs):
        key, _ = key_from_args(*args, cache=True, **kwargs)
        tracer = self._get_or_trace(key)

        def print_stencil(**indices):
            pos = spatial(**indices)
            print(f"{self.f.__name__}: {pos} = {' + '.join(f'{val[indices]} * {vector_add(pos, offset)}' for offset, val in tracer.val.items() if (val[indices] != 0).all)}")

        return print_stencil


def jit_compile_linear(f: Callable[[X], Y]) -> 'LinearFunction[X, Y]':  # TODO add cache control method, e.g. max_traces
    """
    Compile an optimized representation of the linear function `f`.
    For backends that support sparse tensors, a sparse matrix will be constructed for `f`.

    Can be used as a decorator:
    ```python
    @math.jit_compile_linear
    def my_linear_function(x: math.Tensor) -> math.Tensor:
    ```

    Unlike `jit_compile()`, `jit_compile_linear()` can be called during a regular jit compilation.

    See Also:
        `jit_compile()`

    Args:
        f: Function that is linear in its positional arguments.
            All positional arguments must be of type `Tensor` and `f` must return a `Tensor`.
            `f` may be conditioned on keyword arguments.
            However, passing different values for these will cause `f` to be re-traced unless the conditioning arguments are also being traced.

    Returns:
        `LinearFunction` with similar signature and return values as `f`.
    """
    if isinstance(f, JitFunction):
        f = f.f  # cannot trace linear function from jitted version
    return f if isinstance(f, LinearFunction) else LinearFunction(f)


class GradientFunction:

    def __init__(self, f: Callable, wrt: tuple, get_output: bool, jit=False):
        self.f = f
        self.wrt = wrt
        self.get_output = get_output
        self.traces: Dict[SignatureKey, Callable] = {}
        self.recorded_mappings: Dict[SignatureKey, SignatureKey] = {}
        self.jit = jit

    def _trace_grad(self, in_key: SignatureKey, wrt_natives):
        def f_native(*natives, **kwargs):
            PHI_LOGGER.debug(f"Φ-grad: Evaluating gradient of {self.f.__name__}")
            assert not kwargs
            in_tensors = assemble_tensors(natives, in_key.shapes)
            values = assemble_tree(in_key.tree, in_tensors)
            assert isinstance(values, tuple)  # was disassembled from *args
            with functional_derivative_evaluation(order=1):
                result = self.f(*values, **in_key.kwargs)  # Tensor or tuple/list of Tensors
            nest, out_tensors = disassemble_tree(result)
            result_natives, result_shapes = disassemble_tensors(out_tensors)
            self.recorded_mappings[in_key] = SignatureKey(f_native, nest, result_shapes, None, in_key.backend, in_key.tracing)
            return result_natives
        functional_gradient_generator = in_key.backend.jit_compile_grad if self.jit else in_key.backend.functional_gradient
        return functional_gradient_generator(f_native, wrt=wrt_natives, get_output=self.get_output)

    def __call__(self, *args, **kwargs):
        key, natives = key_from_args(*args, cache=True, **kwargs)
        if not key.backend.supports(Backend.functional_gradient):
            if math.default_backend().supports(Backend.functional_gradient):
                warnings.warn(f"Using {math.default_backend()} for gradient computation because {key.backend} does not support functional_gradient()", RuntimeWarning)
                key.backend = math.default_backend()
            else:
                raise AssertionError(f"functional_gradient() not supported by {key.backend}.")
        wrt_tensors = self._track_wrt(args)
        wrt_natives = self._track_wrt_natives(wrt_tensors, disassemble_tree(args)[1])
        if key not in self.traces:
            self.traces[key] = self._trace_grad(key, wrt_natives)
        native_result = self.traces[key](*natives)
        output_key = match_output_signature(key, self.recorded_mappings, self)
        if self.get_output:
            result_shapes = list(output_key.shapes) + [key.shapes[i] for i in wrt_tensors]
            output_tensors = assemble_tensors(native_result, result_shapes)
            output_structure, grad_tuple = assemble_tree((output_key.tree, [key.tree[i] for i in self.wrt]), output_tensors)
            return output_structure, grad_tuple
        else:
            output_tensors = assemble_tensors(native_result, [key.shapes[i] for i in wrt_tensors])
            return assemble_tree([key.tree[i] for i in wrt_tensors], output_tensors)

    def __repr__(self):
        return f"grad({self.f.__name__})"

    @property
    def __name__(self):
        return self.f.__name__

    def _track_wrt(self, args):
        wrt_tensors = []
        for i, arg in enumerate(args):
            _, tensors = disassemble_tree(arg)
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
    Creates a function which computes the gradient of `f`.

    Example:
    ```python
    def loss_function(x, y):
        prediction = f(x)
        loss = math.l2_loss(prediction - y)
        return loss, prediction

    dx, = functional_gradient(loss_function, get_output=False)(x, y)

    (loss, prediction), (dx, dy) = functional_gradient(loss_function,
                                        wrt=(0, 1), get_output=True)(x, y)
    ```

    Functional gradients are implemented for the following backends:

    * PyTorch: [`torch.autograd.grad`](https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad) / [`torch.autograd.backward`](https://pytorch.org/docs/stable/autograd.html#torch.autograd.backward)
    * TensorFlow: [`tf.GradientTape`](https://www.tensorflow.org/api_docs/python/tf/GradientTape)
    * Jax: [`jax.grad`](https://jax.readthedocs.io/en/latest/jax.html#jax.grad)

    When the gradient function is invoked, `f` is called with tensors that track the gradient.
    For PyTorch, `arg.requires_grad = True` for all positional arguments of `f`.

    Args:
        f: Function to be differentiated.
            `f` must return a floating point `Tensor` with rank zero.
            It can return additional tensors which are treated as auxiliary data and will be returned by the gradient function if `return_values=True`.
            All arguments for which the gradient is computed must be of dtype float or complex.
        get_output: Whether the gradient function should also return the return values of `f`.
        wrt: Arguments of `f` with respect to which the gradient should be computed.
            Example: `wrt_indices=[0]` computes the gradient with respect to the first argument of `f`.

    Returns:
        Function with the same arguments as `f` that returns the value of `f`, auxiliary data and gradient of `f` if `get_output=True`, else just the gradient of `f`.
    """
    return GradientFunction(f, wrt, get_output)


class HessianFunction:

    def __init__(self, f: Callable, wrt: tuple, get_output: bool, get_gradient: bool, dim_suffixes: tuple, jit=False):
        assert isinstance(dim_suffixes, tuple) and len(dim_suffixes) == 2
        self.f = f
        self.wrt = wrt
        self.get_output = get_output
        self.get_gradient = get_gradient
        self.dim_suffixes = dim_suffixes
        self.traces: Dict[SignatureKey, Callable] = {}
        self.recorded_mappings: Dict[SignatureKey, SignatureKey] = {}
        self.jit = jit

    def _trace_hessian(self, in_key: SignatureKey, wrt_natives):
        def f_native(*natives, **kwargs):
            PHI_LOGGER.debug(f"Φ-grad: Evaluating gradient of {self.f.__name__}")
            assert not kwargs
            in_tensors = assemble_tensors(natives, in_key.shapes)
            values = assemble_tree(in_key.tree, in_tensors)
            assert isinstance(values, tuple)  # was disassembled from *args
            with functional_derivative_evaluation(order=2):
                result = self.f(*values, **in_key.kwargs)  # Tensor or tuple/list of Tensors
            nest, out_tensors = disassemble_tree(result)
            result_natives, result_shapes = disassemble_tensors(out_tensors)
            self.recorded_mappings[in_key] = SignatureKey(f_native, nest, result_shapes, None, in_key.backend, in_key.tracing)
            return result_natives
        hessian_generator = in_key.backend.jit_compile_hessian if self.jit else in_key.backend.hessian
        return hessian_generator(f_native, wrt=wrt_natives, get_output=self.get_output, get_gradient=self.get_gradient)

    def __call__(self, *args, **kwargs):
        key, natives, batch_shape = key_from_args_pack_batch(*args, cache=True, **kwargs)
        if not key.backend.supports(Backend.functional_gradient):
            if math.default_backend().supports(Backend.functional_gradient):
                warnings.warn(f"Using {math.default_backend()} for gradient computation because {key.backend} does not support functional_gradient()", RuntimeWarning)
                key.backend = math.default_backend()
            else:
                raise AssertionError(f"functional_gradient() not supported by {key.backend}.")
        wrt_tensors: List[int] = self._track_wrt(args)
        wrt_natives: List[int] = self._track_wrt_natives(wrt_tensors, disassemble_tree(args)[1])
        if key not in self.traces:
            self.traces[key] = self._trace_hessian(key, wrt_natives)
        native_result = self.traces[key](*natives)
        assert len(native_result) == 1 + int(self.get_output) + int(self.get_gradient)
        output_key = match_output_signature(key, self.recorded_mappings, self)
        result = ()
        if self.get_output:
            output_tensors = assemble_tensors(native_result[0], output_key.shapes)
            output_tensors = [math.unpack_dims(t, 'batch', batch_shape) for t in output_tensors]
            # output_tensors = [math.reshaped_tensor(n, [batch_shape, *shape.non_batch]) for n, shape in zip(native_result[0], output_key.shapes)]
            result += assemble_tree(output_key.tree, output_tensors),
        if self.get_gradient:
            grad_tensors = assemble_tensors(native_result[int(self.get_output)], [key.shapes[i] for i in wrt_tensors])
            grad_tensors = [math.unpack_dims(t, 'batch', batch_shape) for t in grad_tensors]
            grads = assemble_tree([key.tree[i] for i in wrt_tensors], grad_tensors)
            if len(grads) == 1:
                grads = grads[0]
            result += grads,
        if len(wrt_natives) == 1:
            native_hessian = native_result[-1][0][0]
            hessian_tensor = math.reshaped_tensor(native_hessian, [batch_shape, *self.shape_with_suffixes(key.shapes[0].non_batch, self.dim_suffixes[0]), *self.shape_with_suffixes(key.shapes[0].non_batch, self.dim_suffixes[1])], check_sizes=True)
            result += assemble_tree(key.tree[0], [hessian_tensor]),
        else:
            assert all([t is None for t in key.tree]), "When computing the Hessian w.r.t. multiple tensors, all inputs must be Tensors."
            raise NotImplementedError()
            hessian_tree = [[] for _ in self.wrt]
            for i in range(len(self.wrt)):
                for j in range(len(self.wrt)):
                    native_hessian_ij = native_result[-1][i][j]
                    hessian_tensor_ij = math.reshaped_tensor(native_hessian_ij, [batch_shape, *key.shapes[i].non_batch, *self.dupli_shape(key.shapes[j].non_batch)], check_sizes=True)
                    hessian_tree[i].append(hessian_tensor_ij)
            result += tuple([tuple(col) for col in hessian_tree]),
        return result

    def shape_with_suffixes(self, shape: Shape, suffix: str):
        return shape._with_names([n+suffix for n in shape.names])

    def __repr__(self):
        return f"grad({self.f.__name__})"

    @property
    def __name__(self):
        return self.f.__name__

    def _track_wrt(self, args):
        wrt_tensors = []
        for i, arg in enumerate(args):
            _, tensors = disassemble_tree(arg)
            wrt_tensors.extend([i] * len(tensors))
        return [t_i for t_i, arg_i in enumerate(wrt_tensors) if arg_i in self.wrt]

    @staticmethod
    def _track_wrt_natives(wrt_tensors, values):
        wrt_natives = []
        for i, value in enumerate(values):
            wrt_natives.extend([i] * len(cached(value)._natives()))
        return [n_i for n_i, t_i in enumerate(wrt_natives) if t_i in wrt_tensors]


def hessian(f: Callable, wrt: tuple or list = (0,), get_output=True, get_gradient=True, dim_suffixes=('', '_')) -> Callable:
    """
    *Experimental. This function currently only supports PyTorch and the Hessian can only be computed w.r.t. one argument.*

    Creates a function which computes the Hessian (second derivative) of `f`.

    Example:
    ```python
    def loss_function(x, y):
        prediction = f(x)
        loss = math.l2_loss(prediction - y)
        return loss, prediction

    hess, = functional_gradient(loss_function, get_output=False, get_gradient=False)(x, y)

    (loss, prediction), (dx, dy), ((dx_dx, dx_dy), (dy_dx, dy_dy)) = functional_gradient(loss_function,
                                        wrt=(0, 1), get_output=True)(x, y)
    ```

    When the gradient function is invoked, `f` is called with tensors that track the gradient.
    For PyTorch, `arg.requires_grad = True` for all positional arguments of `f`.

    Args:
        f: Function to be differentiated.
            `f` must return a floating point `Tensor` with rank zero.
            It can return additional tensors which are treated as auxiliary data and will be returned by the gradient function if `return_values=True`.
            All arguments for which the gradient is computed must be of dtype float or complex.
        wrt: Arguments of `f` with respect to which the gradient should be computed.
            Example: `wrt_indices=[0]` computes the gradient with respect to the first argument of `f`.
        get_output: Whether the Hessian function should also return the return values of `f`.
        get_gradient: Whether the Hessian function should also return the gradient of `f`.
        dim_suffixes: `tuple` containing two strings.
            All Non-batch dimensions of the parameters occur twice in the corresponding Hessian.
            To avoid duplicate names, suffixes are added to non-batch dimensions.
            The dimensions from the first derivative computation are appended with `dim_suffixes[0]` and the second ones with `dim_suffixes[1]`.
            This argument has no effect on the dimension names of the gradient if `get_gradient=True`.

    Returns:
        Function with the same arguments as `f` that returns `(f(x), g(x), H(x))` or less depending on `get_output` and `get_gradient`.
    """
    return HessianFunction(f, wrt, get_output, get_gradient, dim_suffixes)


class CustomGradientFunction:

    def __init__(self, f: Callable, gradient: Callable):
        self.f = f
        self.gradient = gradient
        self.traces: Dict[SignatureKey, Callable] = {}
        self.recorded_mappings: Dict[SignatureKey, SignatureKey] = {}

    def _trace(self, in_key: SignatureKey):
        def forward_native(*natives, **kwargs):
            assert not kwargs
            in_tensors = assemble_tensors(natives, in_key.shapes)
            values = assemble_tree(in_key.tree, in_tensors)
            assert isinstance(values, tuple)  # was disassembled from *args
            result = self.f(*values, **in_key.kwargs)  # Tensor or tuple/list of Tensors
            nest, out_tensors = disassemble_tree(result)
            result_natives, result_shapes = disassemble_tensors(out_tensors)
            self.recorded_mappings[in_key] = SignatureKey(forward_native, nest, result_shapes, None, in_key.backend, in_key.tracing)
            return result_natives

        def backward_native(x_natives, y_natives, dy_natives):
            out_key = self.recorded_mappings[in_key]
            # del self.recorded_mappings[in_key]  # this may be required multiple times
            x_tensors = assemble_tensors(x_natives, in_key.shapes)
            y_tensors = assemble_tensors(y_natives, out_key.shapes)
            dy_tensors = assemble_tensors(dy_natives, out_key.shapes)
            x = assemble_tree(in_key.tree, x_tensors)
            assert isinstance(x, tuple)
            y = assemble_tree(out_key.tree, y_tensors)
            dy = assemble_tree(out_key.tree, dy_tensors)
            result = self.gradient(*x, y, dy, **in_key.kwargs)
            assert isinstance(result, (tuple, list)), "Gradient function must return tuple or list"
            result_natives = self.incomplete_tree_to_natives(result, in_key.tree, list(in_key.shapes))
            return result_natives

        forward_native.__name__ = f"forward '{self.f.__name__ if isinstance(self.f, types.FunctionType) else str(self.f)}'"
        backward_native.__name__ = f"{self.gradient.__name__ if isinstance(self.gradient, types.FunctionType) else str(self.gradient)} (of '{self.f.__name__ if isinstance(self.f, types.FunctionType) else str(self.f)}')"

        return in_key.backend.custom_gradient(forward_native, backward_native)

    def __call__(self, *args, **kwargs):
        key, natives = key_from_args(*args, cache=False, **kwargs)
        if not key.backend.supports(Backend.functional_gradient) and not key.backend.supports(Backend.gradients):
            return self.f(*args, **kwargs)  # no need to use custom gradient if gradients aren't supported anyway
        elif not key.backend.supports(Backend.custom_gradient):
            warnings.warn(f"custom_gradient() not supported by {key.backend}. Running function '{self.f.__name__}' as-is.", RuntimeWarning)
            return self.f(*args, **kwargs)
        if key not in self.traces:
            self.traces[key] = self._trace(key)
            if len(self.traces) >= 8:
                warnings.warn(f"{self.__name__} has been traced {len(self.traces)} times. To avoid memory leaks, call {self.f.__name__}.traces.clear(), {self.f.__name__}.recorded_mappings.clear()", RuntimeWarning, stacklevel=2)
        native_result = self.traces[key](*natives)
        output_key = match_output_signature(key, self.recorded_mappings, self)
        output_tensors = assemble_tensors(native_result, output_key.shapes)
        return assemble_tree(output_key.tree, output_tensors)

    def __repr__(self):
        return f"custom_gradient(forward={self.f.__name__}, backward={self.gradient.__name__}, id={id(self)})"

    @property
    def __name__(self):
        return f"custom_grad({self.f.__name__})"

    @staticmethod
    def incomplete_tree_to_natives(incomplete, nest, complete_shapes: List[Shape]) -> list:
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
                natives = []
                for item in nest:
                    natives_item = CustomGradientFunction.incomplete_tree_to_natives(None, item, complete_shapes)
                    natives.extend(natives_item)
                return type(nest)(natives)
            else:
                assert type(nest) == type(incomplete) and len(nest) == len(incomplete)
                natives = []
                for i_item, c_item in zip(incomplete, nest):
                    natives_item = CustomGradientFunction.incomplete_tree_to_natives(i_item, c_item, complete_shapes)
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
                natives_item = CustomGradientFunction.incomplete_tree_to_natives(i_val, n_val, complete_shapes)
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
    """
    Tracer object for linear and affine functions.
    The sparsity pattern is assumed equal for all grid cells and is reflected in `val` (e.g. for a 5-point stencil, `val` has 5 items).
    The Tensors stored in `val` include position-dependent dimensions, allowing for different stencils at different positions.
    Dimensions not contained in any `val` Tensor are treated as independent (batch dimensions).
    """

    def __init__(self, source: Tensor, values_by_shift: dict, shape: Shape, bias: Tensor):
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
        self.source = source
        self.val: Dict[Shape, Tensor] = simplify_add(values_by_shift)
        self.bias = bias
        self._shape = shape
        self._sparse_coo = self._sparse_csr = self._sparse_csc = None

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

    def apply(self, value: Tensor) -> NativeTensor:
        assert value.shape == self.source.shape
        mat = self.get_sparse_matrix().native()
        independent_dims = self.independent_dims
        # TODO slice for missing dimensions
        order_src = concat_shapes(value.shape.only(independent_dims), value.shape.without(independent_dims))
        order_out = concat_shapes(self._shape.only(independent_dims), self._shape.without(independent_dims))
        native_src = value.native(order=order_src.names)
        backend = choose_backend(native_src)
        native_src = backend.reshape(native_src, (order_src.only(independent_dims).volume, order_src.without(independent_dims).volume))
        native_out = backend.matmul(mat, native_src)
        native_out = backend.reshape(native_out, order_out.sizes)
        return NativeTensor(native_out, order_out)

    def get_sparse_coordinate_matrix(self) -> 'SparseMatrixContainer':
        """
        Builds a sparse matrix that represents this linear operation.
        Independent dimensions, those that can be treated as batch dimensions, are recognized automatically and ignored.
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
            vals.append(reshaped_native(values, [*out_shape]))
        cols = np.stack(cols, -1).flatten()
        backend = choose_backend(*vals)
        vals = backend.flatten(backend.stack(vals, -1))
        rows = np.arange(out_shape.volume * len(self.val)) // len(self.val)
        # TODO sort indices?
        self._sparse_coo = SparseMatrixContainer('coo', (out_shape.volume, src_shape.volume),
                                                 set(self.val.keys()), self.dependent_dims,
                                                 NativeTensor(vals, instance(nnz=len(vals))), rows, cols)
        return self._sparse_coo

    def get_sparse_csr_matrix(self) -> 'SparseMatrixContainer':
        """
        Builds a sparse matrix that represents this linear operation.
        Independent dimensions, those that can be treated as batch dimensions, are recognized automatically and ignored.
        """
        if self._sparse_csr is not None:
            return self._sparse_csr
        coo = self.get_sparse_coordinate_matrix()
        idx = np.arange(1, len(coo.values)+1)  # start indexing at 1 since 0 might get removed
        import scipy.sparse
        scipy_csr = scipy.sparse.csr_matrix((idx, (coo.rows, coo.cols)), shape=coo.shape)
        col_indices = scipy_csr.indices
        row_ptr = scipy_csr.indptr
        if coo.values.nnz.size != len(scipy_csr.data):
            warnings.warn("Failed to create CSR matrix because the CSR matrix contains fewer non-zero values than COO. This can happen when the `x` tensor is too small for the stencil.", RuntimeWarning)
            return coo
        values = coo.values.nnz[wrap(scipy_csr.data - 1, instance('nnz'))]  # Change order accordingly
        self._sparse_csr = SparseMatrixContainer('csr', coo.shape, coo.indices_key, coo.src_shape, values, row_ptr, col_indices)
        return self._sparse_csr

    def get_sparse_csc_matrix(self) -> 'SparseMatrixContainer':
        """
        Builds a sparse matrix that represents this linear operation.
        Independent dimensions, those that can be treated as batch dimensions, are recognized automatically and ignored.
        """
        if self._sparse_csc is not None:
            return self._sparse_csc
        coo = self.get_sparse_coordinate_matrix()
        idx = np.arange(1, len(coo.values)+1)  # start indexing at 1 since 0 might get removed
        import scipy.sparse
        scipy_csr = scipy.sparse.csc_matrix((idx, (coo.rows, coo.cols)), shape=coo.shape)
        row_indices = scipy_csr.indices
        col_ptr = scipy_csr.indptr
        if coo.values.nnz.size != len(scipy_csr.data):
            warnings.warn("Failed to create CSR matrix because the CSR matrix contains fewer non-zero values than COO. This can happen when the `x` tensor is too small for the stencil.", RuntimeWarning)
            return coo
        values = coo.values.nnz[wrap(scipy_csr.data - 1, instance('nnz'))]  # Change order accordingly
        self._sparse_csc = SparseMatrixContainer('csc', coo.shape, coo.indices_key, coo.src_shape, values, row_indices, col_ptr)
        return self._sparse_csc

    def get_sparse_matrix(self, matrix_format: str = None) -> 'SparseMatrixContainer':
        if matrix_format is None:
            if self.default_backend.supports(Backend.csc_matrix):
                matrix_format = 'csc'
            elif self.default_backend.supports(Backend.csr_matrix):
                matrix_format = 'csr'
            else:
                matrix_format = 'coo'
        if matrix_format == 'csc':
            return self.get_sparse_csc_matrix()
        if matrix_format == 'csr':
            return self.get_sparse_csr_matrix()
        elif matrix_format == 'coo':
            return self.get_sparse_coordinate_matrix()
        else:
            raise NotImplementedError(f"Unsupported sparse matrix format: '{matrix_format}'")

    @property
    def dependent_dims(self):
        return merge_shapes(*[t.shape for t in self.val.values()])

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
            val[shift] = val_fun(values)
        bias = bias_fun(self.bias)
        return ShiftLinTracer(self.source, val, new_shape, bias)

    def unstack(self, dimension):
        raise NotImplementedError()

    def __neg__(self):
        return ShiftLinTracer(self.source, {shift: -values for shift, values in self.val.items()}, self._shape, -self.bias)

    def _op1(self, native_function):  # only __neg__ is linear
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
                    val_, other_ = math.join_spaces(val, other)
                    values[dim_shift] = operator(val_, other_)
                bias = operator(self.bias, other)
                return ShiftLinTracer(self.source, values, self._shape & other.shape, bias)
            elif op_symbol in '+-':
                bias = operator(self.bias, other)
                return ShiftLinTracer(self.source, self.val, self._shape & other.shape, bias)
            else:
                raise ValueError(f"Unsupported operation encountered while tracing linear function: {native_function}")

    def _tensor_reduce(self,
                       dims: Tuple[str],
                       native_function: Callable,
                       collapsed_function: Callable = lambda inner_reduced, collapsed_dims_to_reduce: inner_reduced,
                       unaffected_function: Callable = lambda value: value):
        if all(dim not in self._shape for dim in dims):
            return unaffected_function(self)
        raise NotImplementedError()

    def _natives(self) -> tuple:
        """
        This function should only be used to determine the compatible backends, this tensor should be regarded as not available.
        """
        return sum([v._natives() for v in self.val.values()], ()) + self.bias._natives()


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


class SparseMatrixContainer:
    """
    This class holds information about a sparse matrix and can be passed as argument of JIT-compiled functions.
    It is typically craeted by a PhiFlow tracer object, such as `ShiftLinTracer`.
    Only the values tensor is variable, the sparsity pattern is fixed.

    TensorFlow doesn't allow native sparse tensors as arguments of JIT-compiled functions.
    """

    def __init__(self,
                 indexing_type: str,
                 shape: tuple,
                 indices_key,
                 src_shape: Shape,
                 values: Tensor,
                 rows, cols,
                 ):
        """

        Args:
            shape: Sparse matrix shape
            indices_key: Low-dimensional representation of the sparsity pattern, typically a set of offsets.
            rows: Row indices
            cols: Column indices
            values: Values
            src_shape: Non-flattened `Shape` of `x` vectors compatible with this matrix.
        """
        assert indexing_type in ('coo', 'csr', 'csc')
        self.indexing_type = indexing_type
        self.shape = shape
        self.indices_key = indices_key
        self.rows = rows
        self.cols = cols
        self.values = values
        self.src_shape = src_shape

    def __eq__(self, other):
        return isinstance(other, SparseMatrixContainer) and \
               self.indexing_type == other.indexing_type and \
               self.indices_key == other.indices_key and \
               self.src_shape == other.src_shape

    def __variable_attrs__(self):
        return 'values',

    def native(self):
        backend = choose_backend(self.rows, self.cols, *self.values._natives())
        if self.indexing_type == 'csc':
            return backend.csc_matrix(self.cols, self.rows, self.values.native(), self.shape)
        if self.indexing_type == 'csr':
            return backend.csr_matrix(self.cols, self.rows, self.values.native(), self.shape)
        if self.indexing_type == 'coo':
            return backend.sparse_coo_tensor((self.rows, self.cols), self.values.native(), self.shape)
        assert False, self.indexing_type


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
                 preprocess_y: Callable = None,
                 preprocess_y_args: tuple = (),
                 gradient_solve: 'Solve[Y, X]' or None = None):
        assert isinstance(method, str)
        self.method: str = method
        """ Optimization method to use. Available solvers depend on the solve function that is used to perform the solve. """
        self.relative_tolerance: Tensor = math.to_float(wrap(relative_tolerance))
        """ Relative tolerance for linear solves only. This must be `0` for minimization problems.
        For systems of equations *f(x)=y*, the final tolerance is `max(relative_tolerance * norm(y), absolute_tolerance)`. """
        self.absolute_tolerance: Tensor = math.to_float(wrap(absolute_tolerance))
        """ Absolut tolerance for optimization problems and linear solves.
        For systems of equations *f(x)=y*, the final tolerance is `max(relative_tolerance * norm(y), absolute_tolerance)`. """
        self.max_iterations: Tensor = math.to_int32(wrap(max_iterations))
        """ Maximum number of iterations to perform before raising a `NotConverged` error is raised. """
        self.x0 = x0
        """ Initial guess for the method, of same type and dimensionality as the solve result.
         This property must be set to a value compatible with the solution `x` before running a method. """
        self.preprocess_y: Callable = preprocess_y
        """ Function to be applied to the right-hand-side vector of an equation system before solving the system.
        This property is propagated to gradient solves by default. """
        self.preprocess_y_args: tuple = preprocess_y_args
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
            self._gradient_solve = Solve(self.method, self.relative_tolerance, self.absolute_tolerance, self.max_iterations, None, self.suppress, self.preprocess_y, self.preprocess_y_args)
        return self._gradient_solve

    def __repr__(self):
        return f"{self.method} with tolerance {self.relative_tolerance} (rel), {self.absolute_tolerance} (abs), max_iterations={self.max_iterations}"

    def __eq__(self, other):
        if not isinstance(other, Solve):
            return False
        if self.method != other.method \
                or (self.absolute_tolerance != other.absolute_tolerance).any \
                or (self.relative_tolerance != other.relative_tolerance).any \
                or (self.max_iterations != other.max_iterations).any \
                or self.preprocess_y is not other.preprocess_y \
                or self.suppress != other.suppress:
            return False
        return self.x0 == other.x0

    def __variable_attrs__(self):
        return 'x0', 'preprocess_y_args'


class SolveInfo(Generic[X, Y]):
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
                 msg: str,
                 solve_time: float):
        # tuple.__new__(SolveInfo, (x, residual, iterations, function_evaluations, converged, diverged))
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
        if not msg and all_available(diverged, converged):
            if self.diverged.any:
                msg = f"Solve diverged within {iterations if iterations is not None else '?'} iterations using {method}."
            elif not self.converged.trajectory[-1].all:
                msg = f"Solve did not converge to rel={solve.relative_tolerance}, abs={solve.absolute_tolerance} within {solve.max_iterations} iterations using {method}. Max residual: {[math.max_(t.trajectory[-1]) for t in disassemble_tree(self.residual)[1]]}"
            else:
                msg = f"Converged within {iterations if iterations is not None else '?'} iterations."
        self.msg = msg
        """ `str`, termination message """
        self.solve_time = solve_time
        """ Time spent in Backend solve function (in seconds) """

    def __repr__(self):
        return self.msg

    def snapshot(self, index):
        return SolveInfo(self.solve, self.x.trajectory[index], self.residual.trajectory[index], self.iterations.trajectory[index], self.function_evaluations.trajectory[index], self.converged.trajectory[index], self.diverged.trajectory[index], self.method, self.msg, self.solve_time)

    def convergence_check(self, only_warn: bool):
        if not all_available(self.diverged, self.converged):
            return
        if self.diverged.any:
            if Diverged not in self.solve.suppress:
                if only_warn:
                    warnings.warn(self.msg, ConvergenceWarning)
                else:
                    raise Diverged(self)
        if not self.converged.trajectory[-1].all:
            if NotConverged not in self.solve.suppress:
                if only_warn:
                    warnings.warn(self.msg, ConvergenceWarning)
                else:
                    raise NotConverged(self)


class ConvergenceException(RuntimeError):
    """
    Base class for exceptions raised when a solve does not converge.

    See Also:
        `Diverged`, `NotConverged`.
    """

    def __init__(self, result: SolveInfo):
        RuntimeError.__init__(self, result.msg)
        self.result: SolveInfo = result
        """ `SolveInfo` holding information about the solve. """


class ConvergenceWarning(RuntimeWarning):
    pass


class NotConverged(ConvergenceException):
    """
    Raised during optimization if the desired accuracy was not reached within the maximum number of iterations.

    This exception inherits from `ConvergenceException`.

    See Also:
        `Diverged`.
    """

    def __init__(self, result: SolveInfo):
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

    def __init__(self, result: SolveInfo):
        ConvergenceException.__init__(self, result)


class SolveTape:
    """
    Used to record additional information about solves invoked via `solve_linear()`, `solve_nonlinear()` or `minimize()`.
    While a `SolveTape` is active, certain performance optimizations and algorithm implementations may be disabled.

    To access a `SolveInfo` of a recorded solve, use
    ```python
    solve = Solve(method, ...)
    with SolveTape() as solves:
        x = math.solve_linear(f, y, solve)
    result: SolveInfo = solves[solve]  # get by Solve
    result: SolveInfo = solves[0]  # get by index
    ```
    """

    def __init__(self, record_trajectories=False):
        """
        Args:
            record_trajectories: When enabled, the entries of `SolveInfo` will contain an additional batch dimension named `trajectory`.
        """
        self.record_trajectories = record_trajectories
        self.solves: List[SolveInfo] = []
        self.solve_ids: List[str] = []

    def __enter__(self):
        _SOLVE_TAPES.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _SOLVE_TAPES.remove(self)

    def _add(self, solve: Solve, trj: bool, result: SolveInfo):
        if any(s.solve.id == solve.id for s in self.solves):
            warnings.warn("SolveTape contains two results for the same solve settings. SolveTape[solve] will return the first solve result.", RuntimeWarning)
        if self.record_trajectories:
            assert trj, "Solve did not record a trajectory."
            self.solves.append(result)
        elif trj:
            self.solves.append(result.snapshot(-1))
        else:
            self.solves.append(result)
        self.solve_ids.append(solve.id)

    def __getitem__(self, item) -> SolveInfo:
        if isinstance(item, int):
            return self.solves[item]
        else:
            assert isinstance(item, Solve)
            solves = [s for s in self.solves if s.solve.id == item.id]
            if len(solves) == 0:
                raise KeyError(f"No solve recorded with key '{item}'.")
            assert len(solves) == 1
            return solves[0]

    def __iter__(self):
        return iter(self.solves)

    def __len__(self):
        return len(self.solves)


_SOLVE_TAPES: List[SolveTape] = []


def minimize(f: Callable[[X], Y], solve: Solve[X, Y]) -> X:
    """
    Finds a minimum of the scalar function *f(x)*.
    The `method` argument of `solve` determines which optimizer is used.
    All optimizers supported by `scipy.optimize.minimize` are supported,
    see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html .
    Additionally a gradient descent solver with adaptive step size can be used with `method='GD'`.

    `math.minimize()` is limited to backends that support `functional_gradient()`, currently PyTorch, TensorFlow and Jax.

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
    assert (solve.relative_tolerance == 0).all, f"relative_tolerance must be zero for minimize() but got {solve.relative_tolerance}"
    assert solve.preprocess_y is None, "minimize() does not allow preprocess_y"
    x0_nest, x0_tensors = disassemble_tree(solve.x0)
    x0_tensors = [to_float(t) for t in x0_tensors]
    backend = choose_backend_t(*x0_tensors, prefer_default=True)
    batch_dims = merge_shapes(*[t.shape for t in x0_tensors]).batch
    x0_natives = []
    for t in x0_tensors:
        t._expand()
        assert t.shape.is_uniform
        x0_natives.append(reshaped_native(t, [batch_dims, t.shape.non_batch], force_expand=True))
    x0_flat = backend.concat(x0_natives, -1)

    def unflatten_assemble(x_flat, additional_dims: Shape = EMPTY_SHAPE):
        i = 0
        x_tensors = []
        for x0_native, x0_tensor in zip(x0_natives, x0_tensors):
            vol = backend.shape(x0_native)[-1]
            flat_native = x_flat[..., i:i + vol]
            x_tensors.append(reshaped_tensor(flat_native, [*additional_dims, batch_dims, x0_tensor.shape.non_batch]))
            i += vol
        x = assemble_tree(x0_nest, x_tensors)
        return x

    def native_function(x_flat):
        x = unflatten_assemble(x_flat)
        if isinstance(x, (tuple, list)):
            y = f(*x)
        else:
            y = f(x)
        _, y_tensors = disassemble_tree(y)
        return y_tensors[0].sum, reshaped_native(y_tensors[0], [batch_dims])

    atol = backend.to_float(reshaped_native(solve.absolute_tolerance, [batch_dims], force_expand=True))
    maxi = backend.to_int32(reshaped_native(solve.max_iterations, [batch_dims], force_expand=True))
    trj = _SOLVE_TAPES and any(t.record_trajectories for t in _SOLVE_TAPES)
    t = time.perf_counter()
    ret = backend.minimize(solve.method, native_function, x0_flat, atol, maxi, trj)
    t = time.perf_counter() - t
    if not trj:
        assert isinstance(ret, SolveResult)
        converged = reshaped_tensor(ret.converged, [batch_dims])
        diverged = reshaped_tensor(ret.diverged, [batch_dims])
        x = unflatten_assemble(ret.x)
        iterations = reshaped_tensor(ret.iterations, [batch_dims])
        function_evaluations = reshaped_tensor(ret.function_evaluations, [batch_dims])
        residual = reshaped_tensor(ret.residual, [batch_dims])
        result = SolveInfo(solve, x, residual, iterations, function_evaluations, converged, diverged, ret.method, ret.message, t)
    else:  # trajectory
        assert isinstance(ret, (tuple, list)) and all(isinstance(r, SolveResult) for r in ret)
        converged = reshaped_tensor(ret[-1].converged, [batch_dims])
        diverged = reshaped_tensor(ret[-1].diverged, [batch_dims])
        x = unflatten_assemble(ret[-1].x)
        x_ = unflatten_assemble(backend.stack([r.x for r in ret]), additional_dims=batch('trajectory'))
        residual = stack([reshaped_tensor(r.residual, [batch_dims]) for r in ret], batch('trajectory'))
        iterations = reshaped_tensor(ret[-1].iterations, [batch_dims])
        function_evaluations = stack([reshaped_tensor(r.function_evaluations, [batch_dims]) for r in ret], batch('trajectory'))
        result = SolveInfo(solve, x_, residual, iterations, function_evaluations, converged, diverged, ret[-1].method, ret[-1].message, t)
    for tape in _SOLVE_TAPES:
        tape._add(solve, trj, result)
    result.convergence_check(False)  # raises ConvergenceException
    return x


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

    if solve.preprocess_y is not None:
        y = solve.preprocess_y(y)

    def min_func(x):
        diff = f(x) - y
        l2 = l2_loss(diff)
        return l2

    rel_tol_to_abs = solve.relative_tolerance * l2_loss(y)
    min_solve = copy_with(solve, absolute_tolerance=rel_tol_to_abs, relative_tolerance=0, preprocess_y=None)
    return minimize(min_func, min_solve)


def solve_linear(f: Callable[[X], Y],
                 y: Y, solve: Solve[X, Y],
                 f_args: tuple or list = (),
                 f_kwargs: dict = None) -> X:
    """
    Solves the system of linear equations *f(x) = y* and returns *x*.
    For maximum performance, compile `f` using `jit_compile_linear()` beforehand.
    Then, an optimized representation of `f` (such as a sparse matrix) will be used to solve the linear system.

    To obtain additional information about the performed solve, use a `SolveTape`.

    The gradient of this operation will perform another linear solve with the parameters specified by `Solve.gradient_solve`.

    See Also:
        `solve_nonlinear()`, `jit_compile_linear()`.

    Args:
        f: Linear function with `Tensor` or `TensorLike` first parameter and return value.
            `f` can have additional arguments.
        y: Desired output of `f(x)` as `Tensor` or `TensorLike`.
        solve: `Solve` object specifying optimization method, parameters and initial guess for `x`.
        f_args: Additional `Tensor` or `TensorLike` arguments to be passed to `f`.
            `f` need not be linear in these arguments.
            Use this instead of lambda function since a lambda will not be recognized as calling a jit-compiled function.
        f_kwargs: Additional keyword arguments to be passed to `f`.
            These arguments can be of any type.

    Returns:
        x: solution of the linear system of equations `f(x) = y` as `Tensor` or `TensorLike`.

    Raises:
        NotConverged: If the desired accuracy was not be reached within the maximum number of iterations.
        Diverged: If the solve failed prematurely.
    """
    y_tree, y_tensors = disassemble_tree(y)
    x0_tree, x0_tensors = disassemble_tree(solve.x0)
    assert len(x0_tensors) == len(y_tensors) == 1, "Only single-tensor linear solves are currently supported"
    backend = choose_backend_t(*y_tensors, *x0_tensors)

    if not all_available(*y_tensors, *x0_tensors):  # jit mode
        f = jit_compile_linear(f) if backend.supports(Backend.sparse_coo_tensor) else jit_compile(f)

    if isinstance(f, LinearFunction) and (backend.supports(Backend.sparse_coo_tensor) or backend.supports(Backend.csr_matrix)):
        matrix, bias = f.sparse_matrix_and_bias(solve.x0, *f_args, **(f_kwargs or {}))
        return _matrix_solve(y - bias, solve, matrix, backend=backend)  # custom_gradient
    else:
        # arg_tree, arg_tensors = disassemble_tree(f_args)
        # arg_tensors = cached(arg_tensors)
        # f_args = assemble_tree(arg_tree, arg_tensors)
        f_args = cached(f_args)
        # x0_tensors = cached(x0_tensors)
        # solve = copy_with(solve, x0=assemble_tree(x0_tree, x0_tensors))
        solve = cached(solve)
        return _function_solve(y, solve, f_args, f_kwargs=f_kwargs or {}, f=f, backend=backend)  # custom_gradient


def _linear_solve_forward(y, solve: Solve, native_lin_op,
                          active_dims: Shape or None, backend: Backend, is_backprop: bool) -> Any:
    PHI_LOGGER.debug(f"Performing linear solve {solve} with backend {backend}")
    if solve.preprocess_y is not None:
        y = solve.preprocess_y(y, *solve.preprocess_y_args)
    y_nest, (y_tensor,) = disassemble_tree(y)
    x0_nest, (x0_tensor,) = disassemble_tree(solve.x0)
    batch_dims = (y_tensor.shape & x0_tensor.shape).without(active_dims)
    x0_native = backend.as_tensor(reshaped_native(x0_tensor, [batch_dims, active_dims], force_expand=True))
    y_native = backend.as_tensor(reshaped_native(y_tensor, [batch_dims, active_dims], force_expand=True))
    rtol = backend.as_tensor(reshaped_native(math.to_float(solve.relative_tolerance), [batch_dims], force_expand=True))
    atol = backend.as_tensor(reshaped_native(solve.absolute_tolerance, [batch_dims], force_expand=True))
    maxi = backend.as_tensor(reshaped_native(solve.max_iterations, [batch_dims], force_expand=True))
    trj = _SOLVE_TAPES and any(t.record_trajectories for t in _SOLVE_TAPES)
    if trj:
        assert all_available(y_tensor, x0_tensor), "Cannot record linear solve in jit mode"
    t = time.perf_counter()
    ret = backend.linear_solve(solve.method, native_lin_op, y_native, x0_native, rtol, atol, maxi, trj)
    t = time.perf_counter() - t
    if not trj:
        assert isinstance(ret, SolveResult)
        converged = reshaped_tensor(ret.converged, [batch_dims])
        diverged = reshaped_tensor(ret.diverged, [batch_dims])
        x = assemble_tree(x0_nest, [reshaped_tensor(ret.x, [batch_dims, active_dims])])
        iterations = reshaped_tensor(ret.iterations, [batch_dims])
        function_evaluations = reshaped_tensor(ret.function_evaluations, [batch_dims])
        if ret.residual is not None:
            residual = assemble_tree(y_nest, [reshaped_tensor(ret.residual, [batch_dims, active_dims])])
        elif _SOLVE_TAPES:
            residual = backend.linear(native_lin_op, ret.x) - y_native
            residual = assemble_tree(y_nest, [reshaped_tensor(residual, [batch_dims, active_dims])])
        else:
            residual = None
        result = SolveInfo(solve, x, residual, iterations, function_evaluations, converged, diverged, ret.method, ret.message, t)
    else:  # trajectory
        assert isinstance(ret, (tuple, list)) and all(isinstance(r, SolveResult) for r in ret), f"Trajectory recording failed: got {type(ret)}"
        converged = reshaped_tensor(ret[-1].converged, [batch_dims])
        diverged = reshaped_tensor(ret[-1].diverged, [batch_dims])
        x = assemble_tree(x0_nest, [reshaped_tensor(ret[-1].x, [batch_dims, active_dims])])
        x_ = assemble_tree(x0_nest, [stack([reshaped_tensor(r.x, [batch_dims, active_dims]) for r in ret], batch('trajectory'))])
        residual = assemble_tree(y_nest, [stack([reshaped_tensor(r.residual, [batch_dims, active_dims]) for r in ret], batch('trajectory'))])
        iterations = reshaped_tensor(ret[-1].iterations, [batch_dims])
        function_evaluations = stack([reshaped_tensor(r.function_evaluations, [batch_dims]) for r in ret], batch('trajectory'))
        result = SolveInfo(solve, x_, residual, iterations, function_evaluations, converged, diverged, ret[-1].method, ret[-1].message, t)
    for tape in _SOLVE_TAPES:
        tape._add(solve, trj, result)
    result.convergence_check(is_backprop and 'TensorFlow' in backend.name)  # raises ConvergenceException
    return x


def attach_gradient_solve(forward_solve: Callable):
    def implicit_gradient_solve(*args, **kwargs):
        y, solve, *matrix, x, dx = args
        grad_solve = solve.gradient_solve
        x0 = grad_solve.x0 if grad_solve.x0 is not None else zeros_like(solve.x0)
        grad_solve_ = copy_with(solve.gradient_solve, x0=x0)
        if 'is_backprop' in kwargs:
            del kwargs['is_backprop']
        dy = solve_with_grad(dx, grad_solve_, *matrix, is_backprop=True, **kwargs)  # this should hopefully result in implicit gradients for higher orders as well
        return (dy, None, *([None] * len(matrix)))  # return grad w.r.t all variable args: (y, x0, Optional[matrix])
    solve_with_grad = custom_gradient(forward_solve, implicit_gradient_solve)
    return solve_with_grad


def _matrix_solve_forward(y, solve: Solve, matrix: SparseMatrixContainer,
                          backend: Backend = None, is_backprop=False):  # kwargs
    matrix_native = matrix.native()
    active_dims = matrix.src_shape
    result = _linear_solve_forward(y, solve, matrix_native, active_dims=active_dims, backend=backend, is_backprop=is_backprop)
    return result  # must return exactly `x` so gradient isn't computed w.r.t. other quantities


_matrix_solve = attach_gradient_solve(_matrix_solve_forward)


def _function_solve_forward(y, solve: Solve, f_args: tuple,
                            f_kwargs: dict = None, f: Callable = None, backend: Backend = None, is_backprop=False):  # kwargs
    y_nest, (y_tensor,) = disassemble_tree(y)
    x0_nest, (x0_tensor,) = disassemble_tree(solve.x0)
    active_dims = (y_tensor.shape & x0_tensor.shape).non_batch  # assumes batch dimensions are not active
    batches = (y_tensor.shape & x0_tensor.shape).batch

    def native_lin_f(native_x, batch_index=None):
        if batch_index is not None and batches.volume > 1:
            native_x = backend.tile(backend.expand_dims(native_x), [batches.volume, 1])
        x = assemble_tree(x0_nest, [reshaped_tensor(native_x, [batches, active_dims] if backend.ndims(native_x) >= 2 else [active_dims], convert=False)])
        y = f(x, *f_args, **f_kwargs)
        _, (y_tensor,) = disassemble_tree(y)
        y_native = reshaped_native(y_tensor, [batches, active_dims] if backend.ndims(native_x) >= 2 else [active_dims])
        if batch_index is not None and batches.volume > 1:
            y_native = y_native[batch_index]
        return y_native

    result = _linear_solve_forward(y, solve, native_lin_f, active_dims=active_dims, backend=backend, is_backprop=is_backprop)
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


def map_types(f: Callable, dims: Shape or tuple or list or str or Callable, dim_type: Callable or str) -> Callable:
    """
    Wraps a function to change the dimension types of its `Tensor` and `TensorLike` arguments.

    Args:
        f: Function to wrap.
        dims: Concrete dimensions or dimension type, such as `spatial` or `batch`.
            These dimensions will be mapped to `dim_type` for all positional function arguments.
        dim_type: Dimension type, such as `spatial` or `batch`.
            `f` will be called with dimensions remapped to this type.

    Returns:
        Function with signature matching `f`.
    """
    def forward_retype(obj, input_types: Shape):
        tree, tensors = disassemble_tree(obj)
        retyped = []
        for t in tensors:
            for dim in t.shape.only(dims):
                t = t.dimension(dim).as_type(dim_type)
                input_types = math.merge_shapes(input_types, dim.with_size(None))
            retyped.append(t)
        return assemble_tree(tree, retyped), input_types

    def reverse_retype(obj, input_types: Shape):
        tree, tensors = disassemble_tree(obj)
        retyped = []
        for t in tensors:
            for dim in t.shape.only(input_types.names):
                t = t.dimension(dim).as_type(input_types.get_type(dim))
            retyped.append(t)
        return assemble_tree(tree, retyped)

    @wraps(f)
    def retyped_f(*args, **kwargs):
        input_types = EMPTY_SHAPE
        retyped_args = []
        for arg in args:
            retyped_arg, input_types = forward_retype(arg, input_types)
            retyped_args.append(retyped_arg)
        output = f(*retyped_args, **kwargs)
        restored_output = reverse_retype(output, input_types)
        return restored_output

    return retyped_f


def map_s2b(f: Callable) -> Callable:
    """ Map spatial dimensions to batch dimensions. Short for `map_types(f, spatial, batch)`. """
    return map_types(f, spatial, batch)
