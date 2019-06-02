from phi.solver.base import *
import logging


class Multigrid(PressureSolver):

    def __init__(self, solvers, autodiff=False):
        """
A multigrid solver first solves the pressure on a lower-resolution grid and successively upsamples and refines it.
On each grid, i, the pressure is calculated using the i-th provided PressureSolver.
The resulting pressure is then upsampled and given as initial guess to the next level.

This approach reduces the number of high-resolution iterations required, especially if the previous solver had a higher accuracy.
        :param solvers: tuple or list of PressureSolvers with length equal to number of grids
        :param autodiff: if True, use autodiff, else use multigrid forward solver for backprop
        """
        if isinstance(solvers, PressureSolver):
            solvers = [solvers] * 2
        PressureSolver.__init__(self, 'Multigrid',
                                supported_devices=solvers[0].supported_devices,
                                supports_guess=solvers[0].supports_guess,
                                supports_loop_counter=np.all([s.supports_loop_counter for s in solvers]),
                                supports_continuous_masks=True)
        assert np.all([s.supports_guess for s in solvers[1:]]), 'solvers must support initial guess'
        self.solvers = solvers
        self.autodiff = autodiff

    def solve(self, divergence, active_mask, fluid_mask, boundaries, pressure_guess):
        if self.autodiff:
            return _mg_solve_forward(divergence, active_mask, fluid_mask, boundaries, pressure_guess, self.solvers)

        def pressure_gradient(op, grad):
            return  _mg_solve_forward(grad, active_mask, fluid_mask, boundaries, None, self.solvers)[0]

        return math.with_custom_gradient(_mg_solve_forward,
                                  [divergence, active_mask, fluid_mask, boundaries, pressure_guess, self.solvers],
                                  pressure_gradient,
                                  input_index=0, output_index=0,
                                  name_base='multigrid_solve')


def _mg_solve_forward(divergence, active_mask, fluid_mask, boundaries, pressure_guess, solvers):
    if active_mask is not None or fluid_mask is not None:
        if not np.all([s.supports_continuous_masks for s in solvers[:-1]]):
            logging.warning(
                "Multigrid solver: There are boundary conditions inside the domain but "
                "not all intermediate solvers support continuous masks")
    div_lvls = [divergence]
    act_lvls = [active_mask]
    fld_lvls = [fluid_mask]
    for grid_i in range(len(solvers) - 1):
        div_lvls.insert(0, downsample2x(div_lvls[0]))
        act_lvls.insert(0, downsample2x(act_lvls[0]) if act_lvls[0] is not None else None)
        fld_lvls.insert(0, downsample2x(fld_lvls[0]) if fld_lvls[0] is not None else None)
        if pressure_guess is not None:
            pressure_guess = downsample2x(pressure_guess)

    iter_list = []
    for i, div in enumerate(div_lvls):
        pressure_guess, iter = solvers[i].solve(div, act_lvls[i], fld_lvls[i], boundaries, pressure_guess)
        iter_list.append(iter)
        if pressure_guess.shape[1] < divergence.shape[1]:
            pressure_guess = upsample2x(pressure_guess) * 2 ** spatial_rank(divergence)

    return pressure_guess, iter_list