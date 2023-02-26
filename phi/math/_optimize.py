import time
import uuid
import warnings
from typing import Callable, Generic, List, TypeVar, Any, Tuple, Union

import numpy
import numpy as np

from .backend import get_precision
from ._shape import EMPTY_SHAPE, Shape, merge_shapes, batch, non_batch, shape, dual, channel, non_dual
from ._magic_ops import stack, copy_with, rename_dims, unpack_dim
from ._sparse import native_matrix, SparseCoordinateTensor, CompressedSparseMatrix
from ._tensors import Tensor, disassemble_tree, assemble_tree, wrap, cached, NativeTensor, layout
from . import _ops as math
from ._ops import choose_backend_t, zeros_like, all_available, reshaped_native, reshaped_tensor, to_float, reshaped_numpy
from ._functional import custom_gradient, LinearFunction, f_name
from .backend import Backend
from .backend._backend import SolveResult, PHI_LOGGER


X = TypeVar('X')
Y = TypeVar('Y')


class Solve(Generic[X, Y]):
    """
    Specifies parameters and stopping criteria for solving a minimization problem or system of equations.
    """

    def __init__(self,
                 method: str or None = 'auto',
                 rel_tol: float or Tensor = None,
                 abs_tol: float or Tensor = None,
                 x0: X or Any = None,
                 max_iterations: int or Tensor = 1000,
                 suppress: tuple or list = (),
                 preprocess_y: Callable = None,
                 preprocess_y_args: tuple = (),
                 gradient_solve: 'Solve[Y, X]' or None = None):
        method = method or 'auto'
        assert isinstance(method, str)
        self.method: str = method
        """ Optimization method to use. Available solvers depend on the solve function that is used to perform the solve. """
        self.rel_tol: Tensor = math.to_float(wrap(rel_tol)) if rel_tol is not None else None
        """Relative tolerance for linear solves only, defaults to 1e-5 for singe precision solves and 1e-12 for double precision solves.
        This must be unset or `0` for minimization problems.
        For systems of equations *f(x)=y*, the final tolerance is `max(rel_tol * norm(y), abs_tol)`. """
        self.abs_tol: Tensor = math.to_float(wrap(abs_tol)) if abs_tol is not None else None
        """ Absolut tolerance for optimization problems and linear solves.
        Defaults to 1e-5 for singe precision solves and 1e-12 for double precision solves.
        For systems of equations *f(x)=y*, the final tolerance is `max(rel_tol * norm(y), abs_tol)`. """
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
        self.id = str(uuid.uuid4())  # not altered by copy_with(), so that the lookup SolveTape[Solve] works after solve has been copied

    @property
    def gradient_solve(self) -> 'Solve[Y, X]':
        """
        Parameters to use for the gradient pass when an implicit gradient is computed.
        If `None`, a duplicate of this `Solve` is created for the gradient solve.

        In any case, the gradient solve information will be stored in `gradient_solve.result`.
        """
        if self._gradient_solve is None:
            self._gradient_solve = Solve(self.method, self.rel_tol, self.abs_tol, None, self.max_iterations, self.suppress, self.preprocess_y, self.preprocess_y_args)
        return self._gradient_solve

    def __repr__(self):
        return f"{self.method} with tolerance {self.rel_tol} (rel), {self.abs_tol} (abs), max_iterations={self.max_iterations}"

    def __eq__(self, other):
        if not isinstance(other, Solve):
            return False
        if self.method != other.method \
                or (self.abs_tol != other.abs_tol).any \
                or (self.rel_tol != other.rel_tol).any \
                or (self.max_iterations != other.max_iterations).any \
                or self.preprocess_y is not other.preprocess_y \
                or self.suppress != other.suppress:
            return False
        return self.x0 == other.x0

    def __variable_attrs__(self):
        return 'x0', 'preprocess_y_args'

    def with_defaults(self, mode: str):
        assert mode in ('solve', 'optimization')
        result = self
        if result.rel_tol is None:
            result = copy_with(result, rel_tol=_default_tolerance() if mode == 'solve' else wrap(0.))
        if result.abs_tol is None:
            result = copy_with(result, abs_tol=_default_tolerance())
        return result


def _default_tolerance():
    if get_precision() == 64:
        return wrap(1e-12)
    elif get_precision() == 32:
        return wrap(1e-5)
    else:
        return wrap(1e-2)


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
                 msg: Tensor,
                 solve_time: float):
        # tuple.__new__(SolveInfo, (x, residual, iterations, function_evaluations, converged, diverged))
        self.solve: Solve[X, Y] = solve
        """ `Solve`, Parameters specified for the solve. """
        self.x: X = x
        """ `Tensor` or `phi.math.magic.PhiTreeNode`, solution estimate. """
        self.residual: Y = residual
        """ `Tensor` or `phi.math.magic.PhiTreeNode`, residual vector for systems of equations or function value for minimization problems. """
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
        if all_available(diverged, converged, iterations):
            msg = math.map_(_default_solve_info_msg, msg, converged.trajectory[-1], diverged.trajectory[-1], iterations.trajectory[-1], solve=solve, method=method, residual=residual)
        self.msg = msg
        """ `str`, termination message """
        self.solve_time = solve_time
        """ Time spent in Backend solve function (in seconds) """

    def __repr__(self):
        return f"{self.method}: {self.converged.trajectory[-1].sum} converged, {self.diverged.trajectory[-1].sum} diverged"

    def snapshot(self, index):
        return SolveInfo(self.solve, self.x.trajectory[index], self.residual.trajectory[index], self.iterations.trajectory[index], self.function_evaluations.trajectory[index],
                         self.converged.trajectory[index], self.diverged.trajectory[index], self.method, self.msg, self.solve_time)

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


def _default_solve_info_msg(msg, converged, diverged, iterations, solve: Solve, method, residual):
    if msg:
        return msg
    if diverged:
        return f"Solve diverged within {iterations if iterations is not None else '?'} iterations using {method}."
    elif not converged:
        max_res = [f"{math.max_(t.trajectory[-1]):no-color:no-dtype}" for t in disassemble_tree(residual)[1]]
        return f"{method} did not converge to rel_tol={float(solve.rel_tol):.0e}, abs_tol={float(solve.abs_tol):.0e} within {int(solve.max_iterations)} iterations. Max residual: {', '.join(max_res)}"
    else:
        return f"Converged within {iterations if iterations is not None else '?'} iterations."


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
    >>> solve = Solve(method, ...)
    >>> with SolveTape() as solves:
    >>>     x = math.solve_linear(f, y, solve)
    >>> result: SolveInfo = solves[solve]  # get by Solve
    >>> result: SolveInfo = solves[0]  # get by index
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

    `math.minimize()` is limited to backends that support `jacobian()`, i.e. PyTorch, TensorFlow and Jax.

    To obtain additional information about the performed solve, use a `SolveTape`.

    See Also:
        `solve_nonlinear()`.

    Args:
        f: Function whose output is subject to minimization.
            All positional arguments of `f` are optimized and must be `Tensor` or `phi.math.magic.PhiTreeNode`.
            If `solve.x0` is a `tuple` or `list`, it will be passed to *f* as varargs, `f(*x0)`.
            To minimize a subset of the positional arguments, define a new (lambda) function depending only on those.
            The first return value of `f` must be a scalar float `Tensor` or `phi.math.magic.PhiTreeNode`.
        solve: `Solve` object to specify method type, parameters and initial guess for `x`.

    Returns:
        x: solution, the minimum point `x`.

    Raises:
        NotConverged: If the desired accuracy was not be reached within the maximum number of iterations.
        Diverged: If the optimization failed prematurely.
    """
    solve = solve.with_defaults('optimization')
    assert (solve.rel_tol == 0).all, f"rel_tol must be zero for minimize() but got {solve.rel_tol}"
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

    def unflatten_assemble(x_flat, additional_dims: Shape = EMPTY_SHAPE, convert=True):
        i = 0
        x_tensors = []
        for x0_native, x0_tensor in zip(x0_natives, x0_tensors):
            vol = backend.shape(x0_native)[-1]
            flat_native = x_flat[..., i:i + vol]
            x_tensors.append(reshaped_tensor(flat_native, [*additional_dims, batch_dims, x0_tensor.shape.non_batch], convert=convert))
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
        assert not non_batch(y_tensors[0]), f"Failed to minimize '{f.__name__}' because it returned a non-scalar output {shape(y_tensors[0])}. Reduce all non-batch dimensions, e.g. using math.l2_loss()"
        try:
            loss_native = reshaped_native(y_tensors[0], [batch_dims])
        except AssertionError:
            raise AssertionError(f"Failed to minimize '{f.__name__}' because its output loss {shape(y_tensors[0])} has more batch dimensions than the initial guess {batch_dims}.")
        return y_tensors[0].sum, (loss_native,)

    atol = backend.to_float(reshaped_native(solve.abs_tol, [batch_dims], force_expand=True))
    maxi = reshaped_numpy(solve.max_iterations, [batch_dims], force_expand=True)
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
        x_ = unflatten_assemble(numpy.stack([r.x for r in ret]), additional_dims=batch('trajectory'), convert=False)
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

    This method is limited to backends that support `jacobian()`, currently PyTorch, TensorFlow and Jax.

    To obtain additional information about the performed solve, use a `SolveTape`.

    See Also:
        `minimize()`, `solve_linear()`.

    Args:
        f: Function whose output is optimized to match `y`.
            All positional arguments of `f` are optimized and must be `Tensor` or `phi.math.magic.PhiTreeNode`.
            The output of `f` must match `y`.
        y: Desired output of `f(x)` as `Tensor` or `phi.math.magic.PhiTreeNode`.
        solve: `Solve` object specifying optimization method, parameters and initial guess for `x`.

    Returns:
        x: Solution fulfilling `f(x) = y` within specified tolerance as `Tensor` or `phi.math.magic.PhiTreeNode`.

    Raises:
        NotConverged: If the desired accuracy was not be reached within the maximum number of iterations.
        Diverged: If the solve failed prematurely.
    """
    def min_func(x):
        diff = f(x) - y
        l2 = l2_loss(diff)
        return l2
    if solve.preprocess_y is not None:
        y = solve.preprocess_y(y)
    from ._nd import l2_loss
    solve = solve.with_defaults('solve')
    tol = math.maximum(solve.rel_tol * l2_loss(y), solve.abs_tol)
    min_solve = copy_with(solve, abs_tol=tol, rel_tol=0, preprocess_y=None)
    return minimize(min_func, min_solve)


def solve_linear(f: Callable[[X], Y] or Tensor,
                 y: Y,
                 solve: Solve[X, Y],
                 *f_args,
                 grad_for_f=False,
                 f_kwargs: dict = None,
                 **f_kwargs_) -> X:
    """
    Solves the system of linear equations *f(x) = y* and returns *x*.
    This method will use the solver specified in `solve`.
    The following method identifiers are supported by all backends:

    * `'auto'`: Automatically choose a solver
    * `'CG'`: Conjugate gradient, only for symmetric and positive definite matrices.
    * `'CG-adaptive'`: Conjugate gradient with adaptive step size, only for symmetric and positive definite matrices.
    * `'biCG'`: Biconjugate gradient
    * `'biCGstab'`: Biconjugate gradient stabilized, first order
    * `'biCGstab(2)'`: Biconjugate gradient stabilized, second order

    For maximum performance, compile `f` using `jit_compile_linear()` beforehand.
    Then, an optimized representation of `f` (such as a sparse matrix) will be used to solve the linear system.

    To obtain additional information about the performed solve, perform the solve within a `SolveTape` context.
    The used implementation can be obtained as `SolveInfo.method`.

    The gradient of this operation will perform another linear solve with the parameters specified by `Solve.gradient_solve`.

    See Also:
        `solve_nonlinear()`, `jit_compile_linear()`.

    Args:
        f: One of the following:

            * Linear function with `Tensor` or `phi.math.magic.PhiTreeNode` first parameter and return value. `f` can have additional auxiliary arguments and return auxiliary values.
            * Dense matrix (`Tensor` with at least one dual dimension)
            * Sparse matrix (Sparse `Tensor` with at least one dual dimension)
            * Native tensor (not yet supported)

        y: Desired output of `f(x)` as `Tensor` or `phi.math.magic.PhiTreeNode`.
        solve: `Solve` object specifying optimization method, parameters and initial guess for `x`.
        *f_args: Positional arguments to be passed to `f` after `solve.x0`. These arguments will not be solved for.
            Supports vararg mode or pass all arguments as a `tuple`.
        f_kwargs: Additional keyword arguments to be passed to `f`.
            These arguments are treated as auxiliary arguments and can be of any type.

    Returns:
        x: solution of the linear system of equations `f(x) = y` as `Tensor` or `phi.math.magic.PhiTreeNode`.

    Raises:
        NotConverged: If the desired accuracy was not be reached within the maximum number of iterations.
        Diverged: If the solve failed prematurely.
    """
    # --- Handle parameters ---
    f_kwargs = f_kwargs or {}
    f_kwargs.update(f_kwargs_)
    f_args = f_args[0] if len(f_args) == 1 and isinstance(f_args[0], tuple) else f_args
    # --- Get input and output tensors ---
    solve = solve.with_defaults('solve')
    y_tree, y_tensors = disassemble_tree(y)
    x0_tree, x0_tensors = disassemble_tree(solve.x0)
    assert solve.x0 is not None, "Please specify the initial guess as Solve(..., x0=initial_guess)"
    assert len(x0_tensors) == len(y_tensors) == 1, "Only single-tensor linear solves are currently supported"
    backend = choose_backend_t(*y_tensors, *x0_tensors)
    prefer_explicit = backend.supports(Backend.sparse_coo_tensor) or backend.supports(Backend.csr_matrix) or grad_for_f

    if isinstance(f, Tensor) or (isinstance(f, LinearFunction) and prefer_explicit):  # Matrix solve
        if isinstance(f, LinearFunction):
            matrix, bias = f.sparse_matrix_and_bias(solve.x0, *f_args, **f_kwargs)
        else:
            matrix = f
            bias = 0

        def _matrix_solve_forward(y, solve: Solve, matrix: Tensor, is_backprop=False):
            backend_matrix = native_matrix(matrix)
            pattern_dims_in = channel(**dual(matrix).untyped_dict).names
            pattern_dims_out = non_dual(matrix).names  # batch dims can be sparse or batched matrices
            result = _linear_solve_forward(y, solve, backend_matrix, pattern_dims_in, pattern_dims_out, backend, is_backprop)
            return result  # must return exactly `x` so gradient isn't computed w.r.t. other quantities

        _matrix_solve = attach_gradient_solve(_matrix_solve_forward, auxiliary_args='is_backprop,solve', matrix_adjoint=grad_for_f)
        return _matrix_solve(y - bias, solve, matrix)
    else:  # Matrix-free solve
        f_args = cached(f_args)
        solve = cached(solve)
        assert not grad_for_f, f"grad_for_f=True can only be used for math.jit_compile_linear functions but got '{f_name(f)}'. Please decorate the linear function with @jit_compile_linear"

        def _function_solve_forward(y, solve: Solve, f_args: tuple, f_kwargs: dict = None, is_backprop=False):
            y_nest, (y_tensor,) = disassemble_tree(y)
            x0_nest, (x0_tensor,) = disassemble_tree(solve.x0)
            # active_dims = (y_tensor.shape & x0_tensor.shape).non_batch  # assumes batch dimensions are not active
            batches = (y_tensor.shape & x0_tensor.shape).batch

            def native_lin_f(native_x, batch_index=None):
                if batch_index is not None and batches.volume > 1:
                    native_x = backend.tile(backend.expand_dims(native_x), [batches.volume, 1])
                x = assemble_tree(x0_nest, [reshaped_tensor(native_x, [batches, non_batch(x0_tensor)] if backend.ndims(native_x) >= 2 else [non_batch(x0_tensor)], convert=False)])
                y = f(x, *f_args, **f_kwargs)
                _, (y_tensor,) = disassemble_tree(y)
                y_native = reshaped_native(y_tensor, [batches, non_batch(y_tensor)] if backend.ndims(native_x) >= 2 else [non_batch(y_tensor)])
                if batch_index is not None and batches.volume > 1:
                    y_native = y_native[batch_index]
                return y_native

            result = _linear_solve_forward(y, solve, native_lin_f, pattern_dims_in=non_batch(x0_tensor).names, pattern_dims_out=non_batch(y_tensor).names, backend=backend, is_backprop=is_backprop)
            return result  # must return exactly `x` so gradient isn't computed w.r.t. other quantities

        _function_solve = attach_gradient_solve(_function_solve_forward, auxiliary_args='is_backprop,f_kwargs,solve', matrix_adjoint=grad_for_f)
        return _function_solve(y, solve, f_args, f_kwargs=f_kwargs)


def _linear_solve_forward(y,
                          solve: Solve,
                          native_lin_op,
                          pattern_dims_in: Tuple[str, ...],
                          pattern_dims_out: Tuple[str, ...],
                          backend: Backend,
                          is_backprop: bool) -> Any:
    PHI_LOGGER.debug(f"Performing linear solve {solve} with backend {backend}")
    if solve.preprocess_y is not None:
        y = solve.preprocess_y(y, *solve.preprocess_y_args)
    y_nest, (y_tensor,) = disassemble_tree(y)
    x0_nest, (x0_tensor,) = disassemble_tree(solve.x0)
    pattern_dims_in = x0_tensor.shape.only(pattern_dims_in)
    pattern_dims_out = y_tensor.shape.only(pattern_dims_out)
    batch_dims = merge_shapes(y_tensor.shape.without(pattern_dims_out), x0_tensor.shape.without(pattern_dims_in))
    x0_native = backend.as_tensor(reshaped_native(x0_tensor, [batch_dims, pattern_dims_in], force_expand=True))
    y_native = backend.as_tensor(reshaped_native(y_tensor, [batch_dims, y_tensor.shape.only(pattern_dims_out)], force_expand=True))
    rtol = backend.as_tensor(reshaped_native(math.to_float(solve.rel_tol), [batch_dims], force_expand=True))
    atol = backend.as_tensor(reshaped_native(solve.abs_tol, [batch_dims], force_expand=True))
    tol_sq = backend.maximum(rtol ** 2 * backend.sum(y_native ** 2, -1), atol ** 2)
    trj = _SOLVE_TAPES and any(t.record_trajectories for t in _SOLVE_TAPES)
    if trj:
        assert all_available(y_tensor, x0_tensor), "Cannot record linear solve in jit mode"
        max_iter = np.expand_dims(np.arange(int(solve.max_iterations)+1), -1)
    else:
        max_iter = reshaped_numpy(solve.max_iterations, [shape(solve.max_iterations).without(batch_dims), batch_dims], force_expand=True)
    t = time.perf_counter()
    ret = backend.linear_solve(solve.method, native_lin_op, y_native, x0_native, tol_sq, max_iter)
    t = time.perf_counter() - t
    trj_dims = [batch(trajectory=len(max_iter))] if trj else []
    assert isinstance(ret, SolveResult)
    converged = reshaped_tensor(ret.converged, [*trj_dims, batch_dims])
    diverged = reshaped_tensor(ret.diverged, [*trj_dims, batch_dims])
    x = assemble_tree(x0_nest, [reshaped_tensor(ret.x, [*trj_dims, batch_dims, pattern_dims_in])])
    iterations = reshaped_tensor(ret.iterations, [*trj_dims, batch_dims])
    function_evaluations = reshaped_tensor(ret.function_evaluations, [*trj_dims, batch_dims])
    if ret.residual is not None:
        residual = assemble_tree(y_nest, [reshaped_tensor(ret.residual, [*trj_dims, batch_dims, pattern_dims_out])])
    elif _SOLVE_TAPES:
        residual = backend.linear(native_lin_op, ret.x) - y_native
        residual = assemble_tree(y_nest, [reshaped_tensor(residual, [*trj_dims, batch_dims, pattern_dims_out])])
    else:
        residual = None
    msg = unpack_dim(layout(ret.message, batch('_all')), '_all', batch_dims)
    result = SolveInfo(solve, x, residual, iterations, function_evaluations, converged, diverged, ret.method, msg, t)
    # else:  # trajectory
    #     converged = reshaped_tensor(ret[-1].converged, [batch_dims])
    #     diverged = reshaped_tensor(ret[-1].diverged, [batch_dims])
    #     x = assemble_tree(x0_nest, [reshaped_tensor(ret[-1].x, [batch_dims, pattern_dims_in])])
    #     x_ = assemble_tree(x0_nest, [stack([reshaped_tensor(r.x, [batch_dims, pattern_dims_in]) for r in ret], )])
    #     residual = assemble_tree(y_nest, [stack([reshaped_tensor(r.residual, [batch_dims, pattern_dims_out]) for r in ret], batch('trajectory'))])
    #     iterations = reshaped_tensor(ret[-1].iterations, [batch_dims])
    #     function_evaluations = stack([reshaped_tensor(r.function_evaluations, [batch_dims]) for r in ret], batch('trajectory'))
    #     result = SolveInfo(solve, x_, residual, iterations, function_evaluations, converged, diverged, ret[-1].method, ret[-1].message, t)
    for tape in _SOLVE_TAPES:
        tape._add(solve, trj, result)
    result.convergence_check(is_backprop and 'TensorFlow' in backend.name)  # raises ConvergenceException
    return x[{'trajectory': -1}] if isinstance(x, Tensor) else x


def attach_gradient_solve(forward_solve: Callable, auxiliary_args: str, matrix_adjoint: bool):
    def implicit_gradient_solve(fwd_args: dict, x, dx):
        solve = fwd_args['solve']
        matrix = (fwd_args['matrix'],) if 'matrix' in fwd_args else ()
        if matrix_adjoint:
            assert matrix, "No matrix given but matrix_gradient=True"
        grad_solve = solve.gradient_solve
        x0 = grad_solve.x0 if grad_solve.x0 is not None else zeros_like(solve.x0)
        grad_solve_ = copy_with(solve.gradient_solve, x0=x0)
        if 'is_backprop' in fwd_args:
            del fwd_args['is_backprop']
        dy = solve_with_grad(dx, grad_solve_, *matrix, is_backprop=True, **fwd_args)  # this should hopefully result in implicit gradients for higher orders as well
        if matrix_adjoint:  # matrix adjoint = dy * x^T sampled at indices
            matrix = matrix[0]
            if isinstance(matrix, CompressedSparseMatrix):
                matrix = matrix.decompress()
            if isinstance(matrix, SparseCoordinateTensor):
                col = matrix.dual_indices(to_primal=True)
                row = matrix.primal_indices()
                dm_values = dy[col] * x[row]
                dm = matrix._with_values(dm_values)
            elif isinstance(matrix, NativeTensor):
                dy_dual = rename_dims(dy, shape(dy), dual(**shape(dy).untyped_dict))
                dm = dy_dual * x  # outer product
                raise NotImplementedError("Matrix adjoint not yet supported for dense matrices")
            else:
                raise AssertionError
            return {'y': dy, 'matrix': dm}
        else:
            return {'y': dy}

    solve_with_grad = custom_gradient(forward_solve, implicit_gradient_solve, auxiliary_args=auxiliary_args)
    return solve_with_grad

