from collections import namedtuple
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


class SolveResult:
    """
    Stores information about the found solution and the performed solving process.
    """

    def __init__(self, iterations: int, **solve_info):
        self.iterations = iterations
        """ Number of iterations performed. """
        self.solve_info = solve_info


class SolveNotConverged(RuntimeError):
    """
    This exception is thrown when a solve did not converge to the specified accuracy.
    """

    def __init__(self, solve: Solve, diverged: bool, msg: str = None):
        if msg is None:
            if diverged:
                msg = f"Solve diverged within {solve.result.iterations} iterations."
            else:
                msg = f"Solve did not converge to rel={solve.relative_tolerance}, abs={solve.absolute_tolerance} within {solve.result.iterations} iterations."
        RuntimeError.__init__(self, msg)
        self.solve: Solve = solve
        """ The specified solve parameters. """
        self.diverged: bool = diverged
        """ Whether the solve stopped because the solution diverged. """
        self.partial_result: SolveResult = solve.result
        """ `SolveResult` containing information about the failed solve. Equivalent to `Solve.result` at the time this exception was created. """
