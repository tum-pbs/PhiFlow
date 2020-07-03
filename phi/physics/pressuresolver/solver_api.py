# coding=utf-8
from phi import math
from phi import struct
from phi.physics.domain import Domain
from phi.physics.field import CenteredGrid
from phi.physics.material import Material
from phi.struct.functions import mappable


class PoissonSolver(object):
    """
    Base class for Poisson solvers
    """

    def __init__(self, name, supported_devices, supports_guess, supports_loop_counter, supports_continuous_masks):
        """Assign details such as name, supported device (CPU/GPU), etc."""
        self.name = name
        self.supported_devices = supported_devices
        self.supports_guess = supports_guess
        self.supports_loop_counter = supports_loop_counter
        self.supports_continuous_masks = supports_continuous_masks

    def solve(self, field, domain, guess, enable_backprop):
        """
        Solves the Poisson equation Δp = d for p for all active fluid cells where active cells are given by the active_mask.
        p is expected to fulfill (Δp-d) ≤ accuracy for every active cell.

        :param enable_backprop: Whether automatic differentiation should be enabled. If False, the gradient of this operation is undefined.
        :param field: scalar input field to the solve, e.g. the divergence of the velocity channel, ∇·v
        :param domain: DomainState object specifying boundary conditions and active/fluid masks. The domain must be equal for all examples (batch dimension equal to 1).
        :param guess: (Optional) Pressure channel which can be used as an initial state for the solver
        :return: pressure tensor (same shape as divergence tensor), number of iterations (integer, 1D integer tensor or None if unknown)
        """
        raise NotImplementedError(self.__class__)

    def __repr__(self):
        """representation = name"""
        return self.name

    def __and__(self, other):
        if isinstance(self, _PoissonSolverChain):
            return _PoissonSolverChain(self.solvers + (other,))
        if isinstance(other, _PoissonSolverChain):
            return _PoissonSolverChain((self,) + other.solvers)
        else:
            return _PoissonSolverChain([self, other])


PressureSolver = PoissonSolver


@struct.definition()
class PoissonDomain(struct.Struct):

    def __init__(self, domain, valid_state=(), active=None, accessible=None, **kwargs):
        struct.Struct.__init__(self, **struct.kwargs(locals(), ignore='valid_state'))
        self._valid_state = valid_state

    @struct.constant()
    def domain(self, domain):
        assert isinstance(domain, Domain), domain
        return domain

    @struct.constant(dependencies='domain')
    def active(self, active):
        extrapolation = _active_extrapolation(Material.extrapolation_mode(self.domain.boundaries))
        if active is not None:
            assert isinstance(active, CenteredGrid)
            assert active.rank == self.domain.rank
            assert active.component_count == 1
            if active.extrapolation != extrapolation:
                active = active.copied_with(extrapolation=extrapolation)
            return active
        else:
            return self.domain.centered_grid(1, extrapolation=extrapolation)

    @struct.constant(dependencies='domain')
    def accessible(self, accessible):
        if accessible is not None:
            assert isinstance(accessible, CenteredGrid)
            assert accessible.rank == self.domain.rank
            assert accessible.component_count == 1
            return accessible
        else:
            return self.domain.centered_grid(1, extrapolation=Material.accessible_extrapolation_mode(self.domain.boundaries))

    @property
    def rank(self):
        return self.domain.rank

    def active_tensor(self, extend=0):
        """
        Scalar channel encoding active cells as ones and inactive (open/obstacle) as zero.
        Active cells are those for which physical constants_dict such as pressure or velocity are calculated.

        :param extend: Extend the grid in all directions beyond the grid size specified by the domain
        """
        return self.active.padded([[extend, extend]] * self.rank).data

    def accessible_tensor(self, extend=0):
        """
        Scalar channel encoding cells that are accessible, i.e. not solid, as ones and obstacles as zero.

        :param extend: Extend the grid in all directions beyond the grid size specified by the domain
        """
        pad_values = struct.map(lambda solid: int(not solid), Material.solid(self.domain.boundaries))
        if isinstance(pad_values, (list, tuple)):
            pad_values = [0] + list(pad_values) + [0]
        result = math.pad(self.accessible.data, [[0,0]] + [[extend, extend]] * self.rank + [[0,0]], constant_values=pad_values)
        return result

    def with_hard_boundary_conditions(self, velocity):
        masked = velocity * self._frictionless_velocity_mask(velocity)
        return masked  # TODO add surface velocity

    def _frictionless_velocity_mask(self, velocity):
        tensors = []
        for axis in range(velocity.rank):
            upper = self.accessible.padded([[0, 1] if ax == axis else [0, 0] for ax in range(self.rank)])
            lower = self.accessible.padded([[1, 0] if ax == axis else [0, 0] for ax in range(self.rank)])
            tensors.append(math.minimum(upper.data, lower.data))
        return velocity.with_data(tensors)


FluidDomain = PoissonDomain


@mappable()
def _active_extrapolation(boundaries):
    return 'periodic' if boundaries == 'periodic' else 'constant'


def poisson_solve(input_field, poisson_domain, solver=None, guess=None, gradient='implicit'):
    """
    Solves the Poisson equation Δp = input_field for p.

    :param gradient: one of ('implicit', 'autodiff', 'inverse')
        If 'autodiff', use the built-in autodiff for backpropagation.
        The intermediate results of each loop iteration will be permanently stored if backpropagation is used.
        If 'implicit', computes a forward pressure solve in reverse accumulation backpropagation.
        This requires less memory but is only accurate if the solution is fully converged.
    :param input_field: CenteredGrid
    :param poisson_domain: PoissonDomain instance
    :param solver: PoissonSolver to use, None for default
    :param guess: CenteredGrid with same size and resolution as input_field
    :return: p as CenteredGrid, iteration count as int or None if not available
    :rtype: CenteredGrid, int
    """
    assert isinstance(input_field, CenteredGrid)
    if guess is not None:
        assert isinstance(guess, CenteredGrid)
        assert guess.compatible(input_field)
        guess = guess.data
    if isinstance(poisson_domain, Domain):
        poisson_domain = PoissonDomain(poisson_domain)
    if solver is None:
        solver = _choose_solver(input_field.resolution, math.choose_backend([input_field.data, poisson_domain.active.data, poisson_domain.accessible.data]))
    if not struct.any(Material.open(poisson_domain.domain.boundaries)):  # has no open boundary
        input_field = input_field - math.mean(input_field.data, axis=tuple(range(1, 1 + input_field.rank)), keepdims=True)  # Subtract mean divergence

    assert gradient in ('autodiff', 'implicit', 'inverse')
    if gradient == 'autodiff':
        pressure, iteration = solver.solve(input_field.data, poisson_domain, guess, enable_backprop=True)
    else:
        if gradient == 'implicit':
            def poisson_gradient(_op, grad):
                return poisson_solve(CenteredGrid.sample(grad, poisson_domain.domain), poisson_domain, solver, None, gradient=gradient)[0].data
        else:  # gradient = 'inverse'
            def poisson_gradient(_op, grad):
                return CenteredGrid.sample(grad, poisson_domain.domain).laplace(physical_units=False).data
        pressure, iteration = math.with_custom_gradient(solver.solve, [input_field.data, poisson_domain, guess, False], poisson_gradient, input_index=0, output_index=0, name_base='poisson_solve')

    pressure = CenteredGrid(pressure, input_field.box, extrapolation=input_field.extrapolation, name='pressure')
    return pressure, iteration


class _PoissonSolverChain(PoissonSolver):

    def __init__(self, solvers):
        PoissonSolver.__init__(self, 'chain%s' % solvers, supported_devices=(), supports_guess=solvers[0].supports_guess, supports_loop_counter=solvers[-1].supports_loop_counter, supports_continuous_masks=False)
        self.solvers = tuple(solvers)
        for solver in solvers[1:]:
            assert isinstance(solver, PoissonSolver)
            assert solver.supports_guess

    def solve(self, field, domain, guess):
        iterations = None
        for solver in self.solvers:
            guess, iterations = solver.solve(field, domain, guess)
        return guess, iterations


def _choose_solver(resolution, backend):
    use_fourier = math.max(resolution) > 64
    if backend.precision == 64:
        from .fourier import FourierSolver
        from .geom import GeometricCG
        return FourierSolver() & GeometricCG(accuracy=1e-8) if use_fourier else GeometricCG(accuracy=1e-8)
    elif backend.precision == 32 and backend.matches_name('SciPy'):
        from .sparse import SparseSciPy
        return SparseSciPy()
    elif backend.precision == 32:
        from .fourier import FourierSolver
        from .sparse import SparseCG
        return FourierSolver() & SparseCG(accuracy=1e-5) if use_fourier else SparseCG(accuracy=1e-5)
    else:  # lower precision
        from .geom import GeometricCG
        return GeometricCG(accuracy=1e-2)
