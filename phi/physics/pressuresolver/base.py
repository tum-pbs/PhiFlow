# coding=utf-8
import numpy as np

from phi import math
from phi import struct
from phi.physics import field


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


class FluidDomain(object):

    def __init__(self, domain, validstate=(), active=None, accessible=None):
        assert active is not None
        assert accessible is not None
        assert domain is not None
        assert np.all(math.staticshape(active) == math.staticshape(accessible))
        assert np.all(domain.resolution == math.staticshape(active)[1:-1])
        self._domain = domain
        self._validstate = validstate
        self._active = active
        self._accessible = accessible

    @property
    def domain(self):
        return self._domain

    @property
    def rank(self):
        return self.domain.rank

    def is_valid(self, state):
        return self._validstate == state

    def with_hard_boundary_conditions(self, velocity):
        masked = velocity * self._frictionless_velocity_mask(velocity)
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

    def _frictionless_velocity_mask(self, velocity):
        """
        """
        tensors = []
        for dim in math.spatial_dimensions(self._accessible):
            upper_pad = 0 if self.domain.surface_material(dim-1, True).solid else 1
            lower_pad = 0 if self.domain.surface_material(dim-1, False).solid else 1
            upper = math.pad(self._accessible, [[0, 1] if d == dim else [0, 0] for d in math.all_dimensions(self._accessible)], constant_values=upper_pad)
            lower = math.pad(self._accessible, [[1, 0] if d == dim else [0, 0] for d in math.all_dimensions(self._accessible)], constant_values=lower_pad)
            tensors.append(math.minimum(upper, lower))
        with struct.anytype():
            components = [field.CenteredGrid(None, tensor) for tensor in tensors]
            data = field.complete_staggered_properties(components, velocity)
        return velocity.with_data(data)
