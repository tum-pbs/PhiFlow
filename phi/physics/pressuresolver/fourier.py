from phi import math
from .solver_api import PoissonSolver


class FourierSolver(PoissonSolver):

    def __init__(self):
        """
        Computes the inverse laplace operation in Fourier space.

        This is computationally inexpensive compared to iterative solvers; the FFT is the most expensive step.

        While the result is only correct for periodic domains, it can be used as initial guess for other solvers, even for non-periodic domains.
        """
        PoissonSolver.__init__(self, 'FFT', ('CPU', 'GPU'), supports_guess=False, supports_loop_counter=False, supports_continuous_masks=False)

    def solve(self, field, domain, guess, enable_backprop):
        return math.fourier_poisson(field), None
