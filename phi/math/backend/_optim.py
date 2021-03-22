from copy import copy


class Solve:
    """
    Specifies parameters and stopping criteria for solving a system of equations or minimization problem.
    """

    def __init__(self,
                 solver: str,
                 relative_tolerance: float,
                 absolute_tolerance: float,
                 max_iterations: int = 1000,
                 gradient_solve: 'Solve' or None = None,
                 **solver_arguments):
        assert isinstance(solver, str)
        self.solver: str = solver
        """ (Optional) Name of method to use. """
        self.relative_tolerance: float = relative_tolerance
        """ The final tolerance is `max(relative_tolerance * norm(y), absolute_tolerance)`. """
        self.absolute_tolerance: float = absolute_tolerance
        """ The final tolerance is `max(relative_tolerance * norm(y), absolute_tolerance)`. """
        self.max_iterations: int = max_iterations
        """ Maximum number of iterations to perform before terminating with `converged=False`. """
        self._gradient_solve = gradient_solve
        if gradient_solve is not None:
            assert gradient_solve.solver == solver
        self.solver_arguments: dict = solver_arguments
        """ Additional solver-dependent arguments. """
        self.result: SolveResult = SolveResult(0)
        """ `SolveResult` storing information about the found solution and the performed solving process. This variable is assigned during the solve. """

    @property
    def gradient_solve(self) -> 'Solve':
        """
        Parameters to use for the spatial_gradient pass when an implicit spatial_gradient is computed. The implicit spatial_gradient must use the same solver.

        If this property is initialized with `None`, its first evaluation will create a duplicate `Solve` object for the spatial_gradient solve.
        Gradient solve information will be stored in `gradient_solve.result`.
        """
        if self._gradient_solve is None:
            self._gradient_solve = copy(self)
        return self._gradient_solve

    def __repr__(self):
        return f"{self.solver} with tolerance {self.relative_tolerance} (rel), {self.absolute_tolerance} (abs), max_iterations={self.max_iterations}"


class SolveResult:
    """
    Stores information about the found solution and the performed solving process.
    """

    def __init__(self, iterations: int, **solve_info):
        self.iterations = iterations
        """ Number of iterations performed. """
        self.solve_info = solve_info

    def __repr__(self):
        return f"[{self.iterations} iterations]"


class ConvergenceException(RuntimeError):
    """
    Base class for exceptions raised when a solve did not converge.

    See Also:
        `Diverged`, `NotConverged`.

    """

    def __init__(self, solve: Solve, x0, x, msg: str):
        # subclasses must have the same signature to be instantiated as type(snc)(...)
        RuntimeError.__init__(self, msg)
        self.solve: Solve = solve
        """ The specified solve parameters as `Solve`. """
        self.partial_result: SolveResult = solve.result
        """ `SolveResult` containing information about the failed solve. Equivalent to `Solve.result` at the time this exception was created. """
        self.x = x
        """ Estimate of result at the end of the failed solve. """
        self.x0 = x0
        """ Initial guess provided to the solver. """

    @property
    def msg(self) -> str:
        """ Human-readable message describing the reason why the optimization failed. """
        return self.args[0]


class NotConverged(ConvergenceException):
    """
    Raised during optimization if the desired accuracy was not reached within the maximum number of iterations.

    This exception inherits from `ConvergenceException`.

    See Also:
        `Diverged`.
    """

    def __init__(self, solve: Solve, x0, x, msg: str = None):
        if msg is None:
            msg = f"Solve did not converge to rel={solve.relative_tolerance}, abs={solve.absolute_tolerance} within {solve.result.iterations} iterations."
        ConvergenceException.__init__(self, solve, x0, x, msg=msg)


class Diverged(ConvergenceException):
    """
    Raised if the optimization was stopped prematurely and cannot continue.
    This may indicate that no solution exists.

    The values of the last estimate `x` may or may not be finite.

    This exception inherits from `ConvergenceException`.

    See Also:
        `NotConverged`.
    """

    def __init__(self, solve: Solve, x0, x, msg: str = None):
        if msg is None:
            msg = f"Solve diverged within {solve.result.iterations} iterations."
        ConvergenceException.__init__(self, solve, x0, x, msg=msg)
