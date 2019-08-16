# coding=utf-8


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

