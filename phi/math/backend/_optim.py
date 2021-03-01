from collections import namedtuple
from copy import copy


class Solve:
    """
    Specifies parameters and stopping criteria for solving a system of equations or minimization problem.
    """

    def __init__(self,
                 solver: str = None,
                 relative_tolerance: float = 1e-5,
                 absolute_tolerance: float = 0,
                 max_iterations: int = 1000,
                 gradient_solve: 'Solve' or None = None,
                 **solver_arguments):
        self.solver = solver
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
        self.result: SolveResult = None
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


class LinearSolve(Solve):
    """
    Specifies parameters and stopping criteria for solving a system of linear equations.

    Extends `Solve` by the property `bake` which determines whether and how the equations are stored.
    """

    def __init__(self,
                 solver: str = None,
                 relative_tolerance=1e-5,
                 absolute_tolerance=0,
                 max_iterations=1000,
                 bake='sparse',
                 gradient_solve: 'Solve' or None = None,
                 **solver_arguments):
        Solve.__init__(self, solver, relative_tolerance, absolute_tolerance, max_iterations, gradient_solve, **solver_arguments)
        self.bake = bake
        """ Baking method: None to use original function, `'sparse'` to create a sparse matrix. """


class SolveResult:
    """
    Stores information about the found solution and the performed solving process.
    """

    def __init__(self, success: bool, iterations: int, **solve_info):
        self.success = success
        """ Whether the solve converged. """
        self.iterations = iterations
        """ Number of iterations performed. """
        self.solve_info = solve_info
