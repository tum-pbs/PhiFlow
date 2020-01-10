# coding=utf-8
from phi import math
from phi import struct
from phi.physics.domain import Domain
from phi.physics.field import CenteredGrid
from phi.physics.material import Material


class PressureSolver(object):
    """
    Base class for solvers
    """

    def __init__(self, name, supported_devices, supports_guess, supports_loop_counter, supports_continuous_masks):
        """Assign details such as name, supported device (CPU/GPU), etc."""
        self.name = name
        self.supported_devices = supported_devices
        self.supports_guess = supports_guess
        self.supports_loop_counter = supports_loop_counter
        self.supports_continuous_masks = supports_continuous_masks

    def solve(self, divergence, domain, pressure_guess):
        """
        Solves the pressure equation Δp = ∇·v for all active fluid cells where active cells are given by the active_mask.
        The resulting pressure is expected to fulfill (Δp-∇·v) ≤ accuracy for every active cell.

        :param divergence: the scalar divergence of the velocity channel, ∇·v
        :param domain: DomainState object specifying boundary conditions and active/fluid masks. The domain must be equal for all examples (batch dimension equal to 1).
        :param pressure_guess: (Optional) Pressure channel which can be used as an initial state for the solver
        :return: pressure tensor (same shape as divergence tensor), number of iterations (integer, 1D integer tensor or None if unknown)
        """
        raise NotImplementedError(self.__class__)

    def __repr__(self):
        """representation = name"""
        return self.name


@struct.definition()
class FluidDomain(struct.Struct):

    def __init__(self, domain, valid_state=(), active=None, accessible=None, **kwargs):
        struct.Struct.__init__(self, **struct.kwargs(locals(), ignore='valid_state'))
        self._valid_state = valid_state

    @struct.constant()
    def domain(self, domain):
        assert isinstance(domain, Domain), domain
        return domain

    @struct.constant(dependencies='domain')
    def active(self, active):
        if active is not None:
            assert isinstance(active, CenteredGrid)
            assert active.rank == self.domain.rank
            assert active.extrapolation == 'constant'
            return active
        else:
            return self.domain.centered_grid(1, extrapolation='constant')

    @struct.constant(dependencies='domain')
    def accessible(self, accessible):
        if accessible is not None:
            assert isinstance(accessible, CenteredGrid)
            assert accessible.rank == self.domain.rank
            return accessible
        else:
            return self.domain.centered_grid(1, extrapolation=Material.extrapolation_mode(self.domain.boundaries))


    @property
    def rank(self):
        return self.domain.rank

    def is_valid(self, state):
        return self._valid_state == state

    def active_tensor(self, extend=0):
        """
        Scalar channel encoding active cells as ones and inactive (open/obstacle) as zero.
        Active cells are those for which physical constants_dict such as pressure or velocity are calculated.

        :param extend: Extend the grid in all directions beyond the grid size specified by the domain
        """
        return math.pad(self.active.data, [[0, 0]] + [[extend, extend]] * self.rank + [[0, 0]], "constant")

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
