import time
import uuid
import warnings
from typing import Callable, Generic, List, TypeVar, Any, Tuple, Union, Optional

import numpy
import numpy as np

from .backend import get_precision, NUMPY
from ._shape import EMPTY_SHAPE, Shape, merge_shapes, batch, non_batch, shape, dual, channel, non_dual, instance, spatial
from ._magic_ops import stack, copy_with, rename_dims, unpack_dim
from ._sparse import native_matrix, SparseCoordinateTensor, CompressedSparseMatrix, stored_values, is_sparse
from ._tensors import Tensor, disassemble_tree, assemble_tree, wrap, cached, NativeTensor, layout
from . import _ops as math
from ._ops import choose_backend_t, zeros_like, all_available, reshaped_native, reshaped_tensor, to_float, reshaped_numpy
from ._functional import custom_gradient, LinearFunction, f_name, _TRACING_JIT
from .backend import Backend
from .backend._backend import SolveResult, PHI_LOGGER, choose_backend, default_backend, convert, Preconditioner
from .backend._linalg import IncompleteLU, incomplete_lu_dense, incomplete_lu_coo, coarse_explicit_preconditioner_coo

X = TypeVar('X')
Y = TypeVar('Y')


class Solve(Generic[X, Y]):
    """
    Specifies parameters and stopping criteria for solving a minimization problem or system of equations.
    """

    def __init__(self,
                 method: Union[str, None] = 'auto',
                 rel_tol: Union[float, Tensor] = None,
                 abs_tol: Union[float, Tensor] = None,
                 x0: Union[X, Any] = None,
                 max_iterations: Union[int, Tensor] = 1000,
                 suppress: Union[tuple, list] = (),
                 preprocess_y: Callable = None,
                 preprocess_y_args: tuple = (),
                 preconditioner: Optional[str] = None,
                 gradient_solve: Union['Solve[Y, X]', None] = None):
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
        self.preconditioner = preconditioner
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
        return f"{self.method} with tolerance {self.rel_tol} (rel), {self.abs_tol} (abs), max_iterations={self.max_iterations}" + (" including preprocessing" if self.preprocess_y else "")

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

    def with_preprocessing(self, preprocess_y: Callable, *args) -> 'Solve':
        """
        Adds preprocessing to this `Solve` and all corresponding gradient solves.

        Args:
            preprocess_y: Preprocessing function.
            *args: Arguments for the preprocessing function.

        Returns:
            Copy of this `Solve` with given preprocessing.
        """
        assert self.preprocess_y is None, f"preprocessing for linear solve '{self}' already set"
        gradient_solve = self._gradient_solve.with_preprocessing(preprocess_y, *args) if self._gradient_solve is not None else None
        return copy_with(self, preprocess_y=preprocess_y, preprocess_y_args=args, _gradient_solve=gradient_solve)


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
                 residual: Union[Y, None],
                 iterations: Union[Tensor, None],
                 function_evaluations: Union[Tensor, None],
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

    def __init__(self, *solves: Solve, record_trajectories=False):
        """
        Args:
            *solves: (Optional) Select specific `solves` to be recorded.
                If none is given, records all solves that occur within the scope of this `SolveTape`.
            record_trajectories: When enabled, the entries of `SolveInfo` will contain an additional batch dimension named `trajectory`.
        """
        self.record_only_ids = [s.id for s in solves]
        self.record_trajectories = record_trajectories
        self.solves: List[SolveInfo] = []

    def should_record_trajectory_for(self, solve: Solve):
        if not self.record_trajectories:
            return False
        if not self.record_only_ids:
            return True
        return solve.id in self.record_only_ids

    def __enter__(self):
        _SOLVE_TAPES.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _SOLVE_TAPES.remove(self)

    def _add(self, solve: Solve, trj: bool, result: SolveInfo):
        if any(s.solve.id == solve.id for s in self.solves):
            warnings.warn("SolveTape contains two results for the same solve settings. SolveTape[solve] will return the first solve result.", RuntimeWarning)
        if self.record_only_ids and solve.id not in self.record_only_ids:
            return  # this solve should not be recorded
        if self.record_trajectories:
            assert trj, "Solve did not record a trajectory."
            self.solves.append(result)
        elif trj:
            self.solves.append(result.snapshot(-1))
        else:
            self.solves.append(result)

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
        x0_natives.append(reshaped_native(t, [batch_dims, t.shape.non_batch]))
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
            loss_native = reshaped_native(y_tensors[0], [batch_dims], force_expand=False)
        except AssertionError:
            raise AssertionError(f"Failed to minimize '{f.__name__}' because its output loss {shape(y_tensors[0])} has more batch dimensions than the initial guess {batch_dims}.")
        return y_tensors[0].sum, (loss_native,)

    atol = backend.to_float(reshaped_native(solve.abs_tol, [batch_dims]))
    maxi = reshaped_numpy(solve.max_iterations, [batch_dims])
    trj = _SOLVE_TAPES and any(t.should_record_trajectory_for(solve) for t in _SOLVE_TAPES)
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


def solve_linear(f: Union[Callable[[X], Y], Tensor],
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
    * `'biCG'` or `'biCG-stab(0)'`: Biconjugate gradient
    * `'biCG-stab'` or `'biCG-stab(1)'`: Biconjugate gradient stabilized, first order
    * `'biCG-stab(2)'`, `'biCG-stab(4)'`, ...: Biconjugate gradient stabilized, second or higher order
    * `'scipy-direct'`: SciPy direct solve always run oh the CPU using `scipy.sparse.linalg.spsolve`.
    * `'scipy-CG'`, `'scipy-GMres'`, `'scipy-biCG'`, `'scipy-biCG-stab'`, `'scipy-CGS'`, `'scipy-QMR'`, `'scipy-GCrotMK'`: SciPy iterative solvers always run oh the CPU.

    **Caution**: SciPy solvers cannot be jit-compiled and should only be used for debugging purposes.

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
        preconditioner = compute_preconditioner(solve.preconditioner, matrix, safe=False, target_backend=NUMPY if solve.method.startswith('scipy-') else backend, solver=solve.method) if solve.preconditioner is not None else None

        def _matrix_solve_forward(y, solve: Solve, matrix: Tensor, is_backprop=False):
            backend_matrix = native_matrix(matrix, choose_backend_t(*y_tensors, matrix))
            pattern_dims_in = channel(**dual(matrix).untyped_dict).names
            pattern_dims_out = non_dual(matrix).names  # batch dims can be sparse or batched matrices
            result = _linear_solve_forward(y, solve, backend_matrix, pattern_dims_in, pattern_dims_out, preconditioner, backend, is_backprop)
            return result  # must return exactly `x` so gradient isn't computed w.r.t. other quantities

        _matrix_solve = attach_gradient_solve(_matrix_solve_forward, auxiliary_args=f'is_backprop,solve{",matrix" if matrix.default_backend == NUMPY else ""}', matrix_adjoint=grad_for_f)
        return _matrix_solve(y - bias, solve, matrix)
    else:  # Matrix-free solve
        f_args = cached(f_args)
        solve = cached(solve)
        assert not grad_for_f, f"grad_for_f=True can only be used for math.jit_compile_linear functions but got '{f_name(f)}'. Please decorate the linear function with @jit_compile_linear"
        assert solve.preconditioner is None, f"Preconditioners not currently supported for matrix-free solves. Decorate '{f_name(f)}' with @math.jit_compile_linear to perform a matrix solve."

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

            result = _linear_solve_forward(y, solve, native_lin_f, pattern_dims_in=non_batch(x0_tensor).names, pattern_dims_out=non_batch(y_tensor).names, preconditioner=None, backend=backend, is_backprop=is_backprop)
            return result  # must return exactly `x` so gradient isn't computed w.r.t. other quantities

        _function_solve = attach_gradient_solve(_function_solve_forward, auxiliary_args='is_backprop,f_kwargs,solve', matrix_adjoint=grad_for_f)
        return _function_solve(y, solve, f_args, f_kwargs=f_kwargs)


def _linear_solve_forward(y,
                          solve: Solve,
                          native_lin_op: Union[Callable, Any],  # native function or native matrix
                          pattern_dims_in: Tuple[str, ...],
                          pattern_dims_out: Tuple[str, ...],
                          preconditioner: Optional[Callable],
                          backend: Backend,
                          is_backprop: bool) -> Any:
    solve = solve.with_defaults('solve')
    PHI_LOGGER.debug(f"Performing linear solve {solve} with backend {backend}")
    if solve.preprocess_y is not None:
        y = solve.preprocess_y(y, *solve.preprocess_y_args)
    y_nest, (y_tensor,) = disassemble_tree(y)
    x0_nest, (x0_tensor,) = disassemble_tree(solve.x0)
    pattern_dims_in = x0_tensor.shape.only(pattern_dims_in, reorder=True)
    pattern_dims_out = y_tensor.shape.only(pattern_dims_out, reorder=True)
    batch_dims = merge_shapes(y_tensor.shape.without(pattern_dims_out), x0_tensor.shape.without(pattern_dims_in))
    x0_native = backend.as_tensor(reshaped_native(x0_tensor, [batch_dims, pattern_dims_in]))
    y_native = backend.as_tensor(reshaped_native(y_tensor, [batch_dims, y_tensor.shape.only(pattern_dims_out)]))
    rtol = backend.as_tensor(reshaped_native(math.to_float(solve.rel_tol), [batch_dims]))
    atol = backend.as_tensor(reshaped_native(solve.abs_tol, [batch_dims]))
    trj = _SOLVE_TAPES and any(t.should_record_trajectory_for(solve) for t in _SOLVE_TAPES)
    if trj:
        max_iter = np.expand_dims(np.arange(int(solve.max_iterations)+1), -1)
    else:
        max_iter = reshaped_numpy(solve.max_iterations, [shape(solve.max_iterations).without(batch_dims), batch_dims])
    t = time.perf_counter()
    ret = backend.linear_solve(solve.method, native_lin_op, y_native, x0_native, rtol, atol, max_iter, preconditioner)
    t = time.perf_counter() - t
    trj_dims = [batch(trajectory=len(max_iter))] if trj else []
    assert isinstance(ret, SolveResult)
    converged = reshaped_tensor(ret.converged, [*trj_dims, batch_dims])
    diverged = reshaped_tensor(ret.diverged, [*trj_dims, batch_dims])
    x = assemble_tree(x0_nest, [reshaped_tensor(ret.x, [*trj_dims, batch_dims, pattern_dims_out])])
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


def compute_preconditioner(method: str, matrix: Tensor, safe=False, target_backend: Backend = None, solver: str = None) -> Optional[Preconditioner]:
    if method == 'auto':
        target_backend = target_backend or default_backend()
        # is_cpu = target_backend.get_default_device().device_type == 'CPU'
        # if tracing and not Backend.supports(Backend.python_call) -> cannot use ILU
        native_triangular = target_backend.supports(Backend.solve_triangular_sparse) if is_sparse(matrix) else target_backend.supports(Backend.solve_triangular_dense)
        if solver in ['direct', 'scipy-direct']:
            method = None
        elif native_triangular:
            method = 'ilu'
        elif spatial(matrix):
            method = 'cluster'
        else:
            method = None
        PHI_LOGGER.info(f"Auto-selecting preconditioner '{method}' for '{solver}' on {target_backend}")
    if method == 'ilu':
        n = dual(matrix).volume
        entry_count = stored_values(matrix).shape.volume
        avg_entries_per_element = entry_count / n
        d = (avg_entries_per_element - 1) / 2
        if _TRACING_JIT and matrix.available:
            iterations = int(math.ceil(n ** (1 / d)))  # high-quality preconditioner when jit-compiling with constant matrix
            PHI_LOGGER.debug(f"factor_ilu: auto-selecting iterations={iterations} (constant matrix) for matrix {matrix}")
        else:
            iterations = int(math.ceil(math.sqrt(d * n ** (1 / d))))  # in 1D take sqrt(n), in 2D take sqrt(2*n**1/2)
            PHI_LOGGER.debug(f"factor_ilu: auto-selecting iterations={iterations} ({'variable matrix' if _TRACING_JIT else 'eager mode'}) for matrix {matrix}")
        lower, upper = factor_ilu(matrix, iterations, safe=safe)
        native_lower = native_matrix(lower, target_backend)
        native_upper = native_matrix(upper, target_backend)
        native_lower = convert(native_lower, target_backend)
        native_upper = convert(native_upper, target_backend)
        return IncompleteLU(native_lower, True, native_upper, False, rank_deficiency=0, source=f"iter={iterations}")  # ToDo rank deficiency
    elif method == 'cluster':
        return explicit_coarse(matrix, target_backend)
    elif method is None:
        return None
    raise NotImplementedError


def factor_ilu(matrix: Tensor, iterations: int, safe=False):
    """
    Incomplete LU factorization for dense or sparse matrices.

    For sparse matrices, keeps the sparsity pattern of `matrix`.
    L and U will be trimmed to the respective areas, i.e. stored upper elements in L will be dropped,
     unless this would lead to varying numbers of stored elements along a batch dimension.

    Args:
        matrix: Dense or sparse matrix to factor.
            Currently, compressed sparse matrices are decompressed before running the ILU algorithm.
        iterations: (Optional) Number of fixed-point iterations to perform.
            If not given, will be automatically determined from matrix size and sparsity.
        safe: If `False` (default), only matrices with a rank deficiency of up to 1 can be factored as all values of L and U are uniquely determined.
            For matrices with higher rank deficiencies, the result includes `NaN` values.
            If `True`, the algorithm runs slightly slower but can factor highly rank-deficient matrices as well.
            However, then L is undeterdetermined and unused values of L are set to 0.
            Rank deficiencies of 1 occur frequently in periodic settings but higher ones are rare.

    Returns:
        L: Lower-triangular matrix as `Tensor` with all diagonal elements equal to 1.
        U: Upper-triangular matrix as `Tensor`.

    Examples:
        >>> matrix = wrap([[-2, 1, 0],
        >>>                [1, -2, 1],
        >>>                [0, 1, -2]], channel('row'), dual('col'))
        >>> L, U = math.factor_ilu(matrix)
        >>> math.print(L)
        row=0      1.          0.          0.         along ~col
        row=1     -0.5         1.          0.         along ~col
        row=2      0.         -0.6666667   1.         along ~col
        >>> math.print(L @ U, "L @ U")
                    L @ U
        row=0     -2.   1.   0.  along ~col
        row=1      1.  -2.   1.  along ~col
        row=2      0.   1.  -2.  along ~col
    """
    if isinstance(matrix, CompressedSparseMatrix):
        matrix = matrix.decompress()
    if isinstance(matrix, SparseCoordinateTensor):
        ind_batch, channels, indices, values, shape = matrix._native_coo_components(dual, matrix=True)
        (l_idx_nat, l_val_nat), (u_idx_nat, u_val_nat) = incomplete_lu_coo(indices, values, shape, iterations, safe)
        col_dims = matrix._shape.only(dual)
        row_dims = matrix._dense_shape.without(col_dims)
        l_indices = matrix._unpack_indices(l_idx_nat[..., 0], l_idx_nat[..., 1], row_dims, col_dims, ind_batch)
        u_indices = matrix._unpack_indices(u_idx_nat[..., 0], u_idx_nat[..., 1], row_dims, col_dims, ind_batch)
        l_values = reshaped_tensor(l_val_nat, [ind_batch, instance(matrix._values), channels], convert=False)
        u_values = reshaped_tensor(u_val_nat, [ind_batch, instance(matrix._values), channels], convert=False)
        lower = SparseCoordinateTensor(l_indices, l_values, matrix._dense_shape, matrix._can_contain_double_entries, matrix._indices_sorted, matrix._default)
        upper = SparseCoordinateTensor(u_indices, u_values, matrix._dense_shape, matrix._can_contain_double_entries, matrix._indices_sorted, matrix._default)
    else:  # dense matrix
        native_matrix = reshaped_native(matrix, [batch, non_batch(matrix).non_dual, dual, EMPTY_SHAPE])
        l_native, u_native = incomplete_lu_dense(native_matrix, iterations, safe)
        lower = reshaped_tensor(l_native, [batch(matrix), non_batch(matrix).non_dual, dual(matrix), EMPTY_SHAPE])
        upper = reshaped_tensor(u_native, [batch(matrix), non_batch(matrix).non_dual, dual(matrix), EMPTY_SHAPE])
    return lower, upper


def explicit_coarse(matrix: Tensor,
                    target_backend: Backend,
                    cluster_count=3 ** 6,
                    cluster_hint=None):
    b0 = matrix.default_backend
    cols = dual(matrix).volume
    # --- cluster entries ---
    if cluster_count >= cols:  # 1 cluster per element
        cluster_count = cols
        clusters = b0.to_int32(b0.linspace_without_last(0, cluster_count, cols))[None, :]
    elif spatial(matrix) and not instance(matrix):  # cell clusters
        axes = spatial(matrix)
        with matrix.default_backend:
            clusters_by_axis = np.round(np.asarray(axes.sizes) * (cluster_count / axes.volume) ** (1/axes.rank)).astype(np.int32)
            cluster_count = int(np.prod(clusters_by_axis))
            clusters_nd = math.meshgrid(axes) / axes * clusters_by_axis
            clusters_nd = math.to_int32(clusters_nd)
            clusters = math.reshaped_native(clusters_nd, [batch, spatial(matrix), 'vector'])
            clusters = b0.ravel_multi_index(clusters, clusters_by_axis)
    else:  # arbitrary clusters
        assert cluster_hint is not None
        raise NotImplementedError(f"Clustering currently only supported for grids but got matrix with shape {matrix.shape}")
    # --- build preconditioner ---
    if isinstance(matrix, CompressedSparseMatrix):
        matrix = matrix.decompress()
    if isinstance(matrix, SparseCoordinateTensor):
        ind_batch, channels, indices, values, shape = matrix._native_coo_components(dual, matrix=True)
        return coarse_explicit_preconditioner_coo(target_backend, indices, values, shape, clusters, cluster_count)
    else:  # dense matrix
        raise NotImplementedError
