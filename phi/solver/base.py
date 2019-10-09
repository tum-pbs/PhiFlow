# coding=utf-8
from phi import struct, math


class PressureSolver(object):

    def __init__(self, name, supported_devices, supports_guess, supports_loop_counter, supports_continuous_masks):
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
        return self.name




class FluidDomain(struct.Struct):
    __struct__ = struct.Def(('_active', 'accessible'))

    def __init__(self, domain, validstate=(), active=None, accessible=None):
        self._domain = domain
        self._validstate = validstate
        self._active = active if active is not None else math.ones(domain.shape())
        self._accessible = accessible if accessible is not None else math.ones(domain.shape())

    @property
    def domain(self):
        return self._domain

    @property
    def rank(self):
        return self.domain.rank

    def is_valid(self, state):
        return self._validstate == state

    def with_hard_boundary_conditions(self, velocity):
        masked = velocity * _frictionless_velocity_mask(self.accessible(extend=1))
        return masked  # TODO add surface velocity

    def active(self, extend=0):
        """
Scalar channel encoding active cells as ones and inactive (open/obstacle) as zero.
Active cells are those for which physical properties such as pressure or velocity are calculated.
        :param extend: Extend the grid in all directions beyond the grid size specified by the domain
        """
        if extend is None or extend == 0:
            return self._active
        else:
            return math.pad(self._active, [[0, 0]] + [[1, 1]] * self.rank + [[0, 0]], "constant")

    def accessible(self, extend=0):
        """
Scalar channel encoding cells that are accessible, i.e. not solid, as ones and obstacles as zero.
        :param extend: Extend the grid in all directions beyond the grid size specified by the domain
        """
        if extend is None or extend == 0:
            return self._accessible
        else:
            solid_paddings, open_paddings = self.domain._get_paddings(lambda material: material.solid, margin=extend)
            mask = self._accessible
            mask = math.pad(mask, open_paddings, "constant", 1)
            mask = math.pad(mask, solid_paddings, "constant", 0)
            return mask

def _frictionless_velocity_mask(accessible_mask):
    dims = range(math.spatial_rank(accessible_mask))
    bcs = []
    for d in dims:
        upper_slices = tuple([(slice(1, None) if i == d else slice(1, None)) for i in dims])
        lower_slices = tuple([(slice(0, -1) if i == d else slice(1, None)) for i in dims])
        bc_d = math.minimum(accessible_mask[(slice(None),) + upper_slices + (slice(None),)],
                            accessible_mask[(slice(None),) + lower_slices + (slice(None),)])
        bcs.append(bc_d)
    return math.StaggeredGrid(math.concat(bcs, axis=-1))