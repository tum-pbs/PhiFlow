import time
import warnings
from dataclasses import dataclass
from typing import Tuple, Callable, Union, Optional

import numpy
import numpy as np
import scipy
from scipy.sparse import issparse, coo_matrix
from scipy.sparse.linalg import spsolve, LinearOperator

from ._backend import Backend, SolveResult, List, DType, spatial_derivative_evaluation, combined_dim, choose_backend, TensorType, Preconditioner, PHI_LOGGER, convert


def pre_str(pre: Optional[Preconditioner]):
    if not pre:
        return ""
    return f"with preconditioner '{pre}'"


def stop_on_l2(b: Backend, rhs_norm_sq, rtol, atol, max_iter: np.ndarray):
    max_iter = b.as_tensor(max_iter[-1, :])
    rsq0 = []
    tol_sq = b.maximum(rtol ** 2 * b.sum(rhs_norm_sq, -1), atol ** 2)
    def check_progress(iterations, residual_squared):
        residual_squared = abs(residual_squared)
        converged = b.all(residual_squared <= tol_sq, axis=(1,))
        if not rsq0:
            diverged = b.any(~b.isfinite(residual_squared), axis=(1,))
            rsq0.append(residual_squared)
        else:
            diverged = b.any(residual_squared / rsq0[0] > 1e5, axis=(1,)) & (iterations >= 8)
            diverged |= b.any(~b.isfinite(residual_squared), axis=(1,))
        continue_ = ~converged & ~diverged & (iterations < max_iter)
        # if on_diverged is not None and b.any(diverged):
        #     on_diverged(iterations)
        return continue_, converged, diverged
    return check_progress


def _max_iter(max_iter: np.ndarray) -> Union[int, list]:
    trj_size, batch_size = max_iter.shape
    if trj_size == 1:
        return int(np.max(max_iter))
    else:
        assert np.all(max_iter == max_iter[:, :1]), "When recording a trajectory, max_iter must be equal for all batch entries"
        return max_iter[:, 0].tolist()


def cg(b: Backend, lin, y, x0, rtol, atol, max_iter, pre: Optional[Preconditioner]) -> Union[SolveResult, List[SolveResult]]:
    """
    Based on "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain" by Jonathan Richard Shewchuk
    symbols: dx=d, dy=q, step_size=alpha, residual_squared=delta, residual=r, y=b, pre=M
    """
    pre = pre or NoPreconditioner()
    batch_size = b.staticshape(y)[0]
    y = b.to_float(y)
    x = b.copy(b.to_float(x0), only_mutable=True)
    residual = y - b.linear(lin, x)
    dx = pre.apply(residual)
    iterations = b.zeros([batch_size], DType(int, 32))
    function_evaluations = b.ones([batch_size], DType(int, 32))
    delta0 = b.sum(residual * dx, -1, keepdims=True)
    check_progress = stop_on_l2(b, abs(delta0), rtol, atol, max_iter)
    continue_, converged, diverged = check_progress(iterations, delta0)

    def cg_loop_body(continue_, x, dx, delta, residual, iterations, function_evaluations, _converged, _diverged):
        continue_1 = b.to_int32(continue_)
        iterations += continue_1
        with spatial_derivative_evaluation(1):
            dy = b.linear(lin, dx); function_evaluations += continue_1
        dx_dy = b.sum(dx * dy, axis=-1, keepdims=True)
        step_size = b.divide_no_nan(delta, dx_dy)
        step_size *= b.expand_dims(b.to_float(continue_1), -1)  # this is not really necessary but ensures batch-independence
        x += step_size * dx
        residual = residual - step_size * dy  # in-place subtraction affects convergence
        s = pre.apply(residual)
        delta_old = delta
        delta = b.sum(residual * s, -1, keepdims=True)
        dx = s + b.divide_no_nan(delta, delta_old) * dx
        continue_, converged, diverged = check_progress(iterations, delta)
        return continue_, x, dx, delta, residual, iterations, function_evaluations, converged, diverged

    _, x, _, _, residual, iterations, function_evaluations, converged, diverged = b.while_loop(cg_loop_body, (continue_, x, dx, delta0, residual, iterations, function_evaluations, converged, diverged), _max_iter(max_iter))
    return SolveResult(f"Φ-Flow CG ({b.name}) {pre_str(pre)}", x, residual, iterations, function_evaluations, converged, diverged, [""] * batch_size)


def cg_adaptive(b, lin, y, x0, rtol, atol, max_iter, pre: Optional[Preconditioner]) -> Union[SolveResult, List[SolveResult]]:
    """
    Based on the variant described in "Methods of Conjugate Gradients for Solving Linear Systems" by Magnus R. Hestenes and Eduard Stiefel https://nvlpubs.nist.gov/nistpubs/jres/049/jresv49n6p409_A1b.pdf
    """
    if pre:
        warnings.warn("CG-adaptive does not yet support preconditioners. Using regular CG instead.", RuntimeWarning)
        return cg(b, lin, y, x0, rtol, atol, max_iter, pre)
    y = b.to_float(y)
    x0 = b.copy(b.to_float(x0), only_mutable=True)
    batch_size = b.staticshape(y)[0]
    x = x0
    dx = residual = y - b.linear(lin, x)
    dy = b.linear(lin, dx)
    iterations = b.zeros([batch_size], DType(int, 32))
    function_evaluations = b.ones([batch_size], DType(int, 32))
    residual_squared = b.sum(residual ** 2, -1, keepdims=True)
    check_progress = stop_on_l2(b, b.sum(y ** 2, -1), rtol, atol, max_iter)
    continue_, converged, diverged = check_progress(iterations, residual_squared)

    def acg_loop_body(continue_, x, dx, dy, residual, iterations, function_evaluations, _converged, _diverged):
        continue_1 = b.to_int32(continue_)
        iterations += continue_1
        dx_dy = b.sum(dx * dy, axis=-1, keepdims=True)
        step_size = b.divide_no_nan(b.sum(dx * residual, axis=-1, keepdims=True), dx_dy)
        step_size *= b.expand_dims(b.to_float(continue_1), -1)  # this is not really necessary but ensures batch-independence
        x += step_size * dx
        residual = residual - step_size * dy  # in-place subtraction affects convergence
        residual_squared = b.sum(residual ** 2, -1, keepdims=True)
        dx = residual - b.divide_no_nan(b.sum(residual * dy, axis=-1, keepdims=True) * dx, dx_dy)
        with spatial_derivative_evaluation(1):
            dy = b.linear(lin, dx); function_evaluations += continue_1
        continue_, converged, diverged = check_progress(iterations, residual_squared)
        return continue_, x, dx, dy, residual, iterations, function_evaluations, converged, diverged

    _, x, _, _, residual, iterations, function_evaluations, converged, diverged = b.while_loop(acg_loop_body, (continue_, x, dx, dy, residual, iterations, function_evaluations, converged, diverged), _max_iter(max_iter))
    return SolveResult(f"Φ-Flow CG-adaptive ({b.name}) {pre_str(pre)}", x, residual, iterations, function_evaluations, converged, diverged, [""] * batch_size)


def bicg(b: Backend, lin, y, x0, rtol, atol, max_iter, pre: Optional[Preconditioner], poly_order: int) -> Union[SolveResult, List[SolveResult]]:
    """ Adapted from [BiCGstab for linear equations involving unsymmetric matrices with complex spectrum](https://dspace.library.uu.nl/bitstream/handle/1874/16827/sleijpen_93_bicgstab.pdf) """
    # Based on "BiCG-stab(L) for linear equations involving asymmetric matrices with complex spectrum" by Gerard L.G. Sleijpen
    if poly_order == 0:
        raise NotImplementedError(f"Generic non-stabilized biCG not supported. Use 'scipy-biCG' instead")
    if poly_order == 1:
        return bicg_stab_first_order(b, lin, y, x0, rtol, atol, max_iter, pre)
    if pre:
        warnings.warn(f"Φ-Flow biCG-stab({poly_order}) with preconditioner is experimental and may diverge.", RuntimeWarning)
    pre = pre or NoPreconditioner()
    y = b.to_float(y)
    x = b.copy(b.to_float(x0), only_mutable=True)
    batch_size = b.staticshape(y)[0]
    r0_tild = residual = y - b.linear(lin, x)
    iterations = b.zeros([batch_size], DType(int, 32))
    function_evaluations = b.ones([batch_size], DType(int, 32))
    residual_squared = b.sum(residual ** 2, -1, keepdims=True)
    check_progress = stop_on_l2(b, b.sum(y ** 2, -1), rtol, atol, max_iter)
    continue_, converged, diverged = check_progress(iterations, residual_squared)
    rho_0 = b.ones([batch_size, 1])
    rho_1 = b.ones([batch_size, 1])
    omega = b.ones([batch_size, 1])
    alpha = b.zeros([batch_size, 1])
    u = b.zeros_like(x)
    r0_hat = [b.zeros(x0.shape)] * (poly_order + 1)
    u_hat = [b.zeros(x0.shape)] * (poly_order + 1)

    def loop_body(continue_, x, residual, iterations, function_evaluations, _converged, _diverged, rho_0, rho_1, omega, alpha, u, u_hat, r0_hat):
        tau = [[b.zeros((batch_size,))] * (poly_order + 1)] * (poly_order + 1)
        sigma = [b.zeros((batch_size,))] * (poly_order + 1)
        gamma = [b.zeros((batch_size,))] * (poly_order + 1)
        gamma_p = [b.zeros((batch_size,))] * (poly_order + 1)
        gamma_pp = [b.zeros((batch_size,))] * (poly_order + 1)
        continue_1 = b.to_int32(continue_)
        iterations += continue_1
        u_hat[0] = u
        r0_hat[0] = residual
        rho_0 = -omega * rho_0
        # --- Bi-CG part ---
        for j in range(0, poly_order):
            rho_1 = b.sum(r0_hat[j] * r0_tild, axis=-1, keepdims=True)
            beta = alpha * rho_1 / rho_0
            rho_0 = rho_1
            for i in range(0, j + 1):
                u_hat[i] = beta * u_hat[i]
                u_hat[i] = r0_hat[i] - u_hat[i]
            put = pre.apply(u_hat[j])
            u_hat[j + 1] = b.linear(lin, put); function_evaluations += continue_1
            gamma_coeff = b.sum(u_hat[j + 1] * r0_tild, axis=-1, keepdims=True)
            alpha = rho_0 / gamma_coeff  # ToDo produces NaN if pre is perfect
            for i in range(0, j + 1):
                r0_hat[i] = r0_hat[i] - alpha * u_hat[i + 1]
            prt = pre.apply(r0_hat[j])
            r0_hat[j + 1] = b.linear(lin, prt); function_evaluations += continue_1
            x = x + alpha * u_hat[0]
        for j in range(1, poly_order + 1):
            for i in range(1, j):
                tau[i][j] = b.sum(r0_hat[j] * r0_hat[i], axis=-1, keepdims=True) / sigma[i]
                r0_hat[j] = r0_hat[j] - tau[i][j] * r0_hat[i]
            sigma[j] = b.sum(r0_hat[j] * r0_hat[j], axis=-1, keepdims=True)
            gamma_p[j] = b.sum(r0_hat[0] * r0_hat[j], axis=-1, keepdims=True) / sigma[j]
        # --- MR part ---
        omega = gamma[poly_order] = gamma_p[poly_order]
        for j in range(poly_order - 1, 0, -1):
            sumg = b.zeros_like(tau[0][0])
            for i in range(j + 1, poly_order + 1):
                sumg = sumg + tau[j][i] * gamma[i]
            gamma[j] = gamma_p[j] - sumg
        for j in range(1, poly_order):
            sumg = b.zeros_like(tau[0][0])
            for i in range(j + 1, poly_order):
                sumg = sumg + tau[j][i] * gamma[i + 1]
            gamma_pp[j] = gamma[j + 1] + sumg
        # --- Update ---
        x = x + gamma[1] * r0_hat[0]
        r0_hat[0] = r0_hat[0] - gamma_p[poly_order] * r0_hat[poly_order]
        u_hat[0] = u_hat[0] - gamma[poly_order] * u_hat[poly_order]
        for j in range(1, poly_order):
            u_hat[0] = u_hat[0] - gamma[j] * u_hat[j]
            x = x + gamma_pp[j] * r0_hat[j]
            r0_hat[0] = r0_hat[0] - gamma_p[j] * r0_hat[j]
        u = u_hat[0]
        residual = r0_hat[0]
        residual_squared = b.sum(residual ** 2, -1, keepdims=True)
        continue_, converged, diverged = check_progress(iterations, residual_squared)
        # ToDo multiply step_size by continue_1 to avoid NaN when iterating after convergence
        return continue_, x, residual, iterations, function_evaluations, converged, diverged, rho_0, rho_1, omega, alpha, u, u_hat, r0_hat

    _, x, residual, iterations, function_evaluations, converged, diverged, rho_0, rho_1, omega, alpha, u, u_hat, r0_hat = b.while_loop(loop_body, (continue_, x, residual, iterations, function_evaluations, converged, diverged, rho_0, rho_1, omega, alpha, u, u_hat, r0_hat), _max_iter(max_iter))
    return SolveResult(f"Φ-Flow biCG-stab({poly_order}) ({b.name}) {pre_str(pre)}", x, residual, iterations, function_evaluations, converged, diverged, [""] * batch_size)


def bicg_stab_first_order(b: Backend, lin, y, x0, rtol, atol, max_iter, pre: Optional[Preconditioner]) -> SolveResult or List[SolveResult]:
    """
    https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method

    L=1, """
    pre = pre or NoPreconditioner()
    y = b.to_float(y)
    x = b.copy(b.to_float(x0), only_mutable=True)
    batch_size = b.staticshape(y)[0]
    residual = y - b.linear(lin, x)
    r0_h = b.ones(x0.shape)
    iterations = b.zeros([batch_size], DType(int, 32))
    function_evaluations = b.ones([batch_size], DType(int, 32))
    residual_squared = b.sum(residual ** 2, -1, keepdims=True)
    check_progress = stop_on_l2(b, b.sum(y ** 2, -1), rtol, atol, max_iter)
    continue_, converged, diverged = check_progress(iterations, residual_squared)
    rho_prev = b.ones([batch_size, 1])
    rho = b.ones([batch_size, 1])
    omega = b.ones([batch_size, 1])
    alpha = b.zeros([batch_size, 1])
    u = b.zeros(x0.shape)
    p = b.zeros(x0.shape)

    def loop_body(continue_, x, residual, iterations, function_evaluations, _converged, _diverged, rho_prev, rho, omega, alpha, u, p):
        continue_1 = b.to_int32(continue_)
        iterations += continue_1
        rho = b.sum(r0_h * residual, axis=-1, keepdims=True)
        beta = rho / rho_prev * alpha / omega;          rho_prev = rho
        p = residual + beta * (p - omega * u)
        y = pre.apply(p)
        u = b.linear(lin, y);                           function_evaluations += continue_1
        alpha = rho / b.sum(r0_h * u, axis=-1, keepdims=True)
        h = x + alpha * y
        s = residual - alpha * u
        s_pre = pre.apply_inv_l(s)
        z = pre.apply(s)
        t = b.linear(lin, z);                         function_evaluations += continue_1
        t_pre = pre.apply_inv_l(t)
        omega = b.sum(t_pre * s_pre, axis=-1, keepdims=True) / b.sum(t_pre * t_pre, axis=-1, keepdims=True)
        x = h + omega * z
        residual = s - omega * t
        residual_squared = b.sum(residual ** 2, -1, keepdims=True)
        continue_, converged, diverged = check_progress(iterations, residual_squared)
        # ToDo multiply step_size by continue_1 to avoid NaN when iterating after convergence
        return continue_, x, residual, iterations, function_evaluations, converged, diverged, rho_prev, rho, omega, alpha, u, p

    _, x, residual, iterations, function_evaluations, converged, diverged, rho_prev, rho, omega, alpha, u, p = b.while_loop(loop_body, (continue_, x, residual, iterations, function_evaluations, converged, diverged, rho_prev, rho, omega, alpha, u, p), _max_iter(max_iter))
    return SolveResult(f"Φ-Flow biCG-stab {pre_str(pre)}", x, residual, iterations, function_evaluations, converged, diverged, [""] * batch_size)


def scipy_spsolve(b: Backend, method: Union[str, Callable], lin, y, x0, rtol, atol, max_iter, pre: Optional[Preconditioner]) -> SolveResult:
    assert max_iter.shape[0] == 1, f"Trajectory recording not supported for scipy_spsolve"
    if method == 'direct':
        if pre:
            warnings.warn(f"Preconditioner {pre} was computed but is not used by SciPy direct solve.", RuntimeWarning)
        return scipy_direct_linear_solve(b, lin, y)
    else:
        if isinstance(method, str):
            function = {
                'CG': scipy.sparse.linalg.cg,
                'GMres': scipy.sparse.linalg.gmres,
                'biCG': scipy.sparse.linalg.bicg,
                'biCG-stab': scipy.sparse.linalg.bicgstab,
                'CGS': scipy.sparse.linalg.cgs,
                'lGMres': scipy.sparse.linalg.lgmres,
                'QMR': scipy.sparse.linalg.qmr,
                'GCrotMK': scipy.sparse.linalg.gcrotmk,
                # 'minres': scipy.sparse.linalg.minres,  # this does not work like the others
            }[method]
        else:
            function = method
        return scipy_iterative_sparse_solve(b, lin, y, x0, rtol, atol, max_iter, pre, function)


def scipy_direct_linear_solve(b: Backend, lin, y):
    batch_size = b.staticshape(y)[0]
    xs = []
    converged = []
    residuals = []
    if isinstance(lin, (tuple, list)):
        assert all(issparse(l) for l in lin)
    else:
        assert issparse(lin)
        lin = [lin] * batch_size
    # Solve each example independently
    for batch in range(batch_size):
        # use_umfpack=self.precision == 64
        x = spsolve(lin[batch], y[batch])  # returns nan when diverges
        xs.append(x)
        converged.append(np.all(np.isfinite(x)))
        residuals.append(lin[batch] @ x - y[batch])
    x = np.stack(xs)
    converged = np.stack(converged)
    residual = np.stack(residuals)
    diverged = ~converged
    iterations = [-1] * batch_size  # spsolve does not perform iterations
    return SolveResult('scipy.sparse.linalg.spsolve', x, residual, iterations, iterations, converged, diverged, [""] * batch_size)


def scipy_iterative_sparse_solve(b: Backend, lin, y, x0, rtol, atol, max_iter, pre, scipy_function: Callable) -> SolveResult:
    if max_iter.shape[0] > 1:
        raise RuntimeError(f"SciPy's sparse solvers (like {scipy_function.__name__}) do not record trajectories. Use a different solver instead.")
    bs_y = b.staticshape(y)[0]
    bs_x0 = b.staticshape(x0)[0]
    batch_size = combined_dim(bs_y, bs_x0)
    # if callable(A):
    #     A = LinearOperator(dtype=y.dtype, shape=(self.staticshape(y)[-1], self.staticshape(x0)[-1]), matvec=A)

    def count_callback(x_n):  # called after each step, not with x0
        iterations[b] += 1

    xs = []
    iterations = [0] * batch_size
    converged = []
    diverged = []
    residual = []
    messages = []
    for b in range(batch_size):
        lin_b = lin[min(b, len(lin)-1)] if isinstance(lin, (tuple, list)) or (isinstance(lin, np.ndarray) and len(lin.shape) > 2) else lin
        pre_op = LinearOperator(shape=lin_b.shape, matvec=pre.apply, rmatvec=pre.apply_transposed) if isinstance(pre, Preconditioner) else None
        x, ret_val = scipy_function(lin_b, y[b], x0=x0[b], tol=rtol[b], atol=atol[b], maxiter=max_iter[-1, b], M=pre_op, callback=count_callback)
        # ret_val: 0=success, >0=not converged, <0=error
        messages.append(f"code {ret_val} (SciPy {scipy_function.__name__})")
        xs.append(x)
        converged.append(ret_val == 0)
        diverged.append(ret_val < 0 or np.any(~np.isfinite(x)))
        residual.append(lin_b @ x - y[b])
    x = np.stack(xs)
    residual = np.stack(residual)
    iterations = np.stack(iterations)
    converged = np.stack(converged)
    diverged = np.stack(diverged)
    return SolveResult(f'scipy.sparse.linalg.{scipy_function.__name__}', x, residual, iterations, iterations + 1, converged, diverged, messages)


class NoPreconditioner(Preconditioner):

    def apply(self, vec):
        return vec

    def apply_transposed(self, vec):
        return vec

    def apply_inv_l(self, vec):
        return vec

    def apply_inv_u(self, vec):
        return vec


@dataclass
class IncompleteLU(Preconditioner):

    lower: TensorType  # (batch_size, rows, cols)
    lower_unit_diagonal: bool
    upper: TensorType  # (batch_size, rows, cols)
    upper_unit_diagonal: bool
    rank_deficiency: int
    source: str

    def __post_init__(self):  # ToDo this is temporary until backend.solve_triangular supports sparse matrices
        # assert choose_backend(self.lower).ndims(self.lower) == 3
        # assert choose_backend(self.upper).ndims(self.lower) == 3
        self._np_lower = choose_backend(self.lower).numpy(self.lower)
        self._np_upper = choose_backend(self.upper).numpy(self.upper)

    def apply_inv_l(self, vec):
        b = choose_backend(self.lower, self.upper, vec)
        return b.solve_triangular(self.lower, vec, lower=True, unit_diagonal=self.lower_unit_diagonal)

    def apply_inv_u(self, vec):
        b = choose_backend(self.lower, self.upper, vec)
        return b.solve_triangular(self.upper, vec, lower=False, unit_diagonal=self.upper_unit_diagonal)

    def apply(self, vec):
        b = choose_backend(vec)
        np_vec = vec if isinstance(vec, numpy.ndarray) else b.numpy(vec)
        from scipy.sparse.linalg import spsolve_triangular
        np_intermediate = spsolve_triangular(self._np_lower, np_vec.T, lower=True, unit_diagonal=self.lower_unit_diagonal)
        np_result = spsolve_triangular(self._np_upper, np_intermediate, lower=False, unit_diagonal=self.upper_unit_diagonal).T
        return np_result if isinstance(vec, numpy.ndarray) else b.as_tensor(np_result)
        # intermediate = b.solve_triangular(self.lower, vec, lower=True, unit_diagonal=self.lower_unit_diagonal)
        # # ToDo if set last rank_deficiency entries to 0, then solve smaller system
        # result = b.solve_triangular(self.upper, intermediate, lower=False, unit_diagonal=self.upper_unit_diagonal)
        # # return result.T
        # return result

    def apply_transposed(self, vec):
        b = choose_backend(self.lower, self.upper, vec)
        np_vec = vec if isinstance(vec, numpy.ndarray) else b.numpy(vec)
        from scipy.sparse.linalg import spsolve_triangular, spsolve
        np_intermediate = spsolve_triangular(self._np_upper.T, np_vec.T, lower=True, unit_diagonal=self.upper_unit_diagonal)
        np_result = spsolve_triangular(self._np_lower.T, np_intermediate, lower=False, unit_diagonal=self.lower_unit_diagonal).T
        return np_result if isinstance(vec, numpy.ndarray) else b.as_tensor(np_result)
        # intermediate = b.solve_triangular(self.upper.T, vec.T, lower=True, unit_diagonal=self.upper_unit_diagonal)
        # result = b.solve_triangular(self.lower.T, intermediate, lower=False, unit_diagonal=self.lower_unit_diagonal).T
        # return result

    def __repr__(self):
        return f"ilu ({self.source})"


def incomplete_lu_dense(matrix, iterations: int, safe: bool):
    """

    Args:
        matrix: Square matrix of Shape (batch_size, rows, cols, channels)
        iterations: Number of fixed-point iterations to perform.
        safe: Avoid NaN when the rank deficiency of `matrix` is 2 or higher.
            For a rank deficiency of 1, the fixed-point algorithm will still converge without NaNs and all values of L and U are uniquely determined.
            If enabled, the algorithm is slightly slower.
            Rank deficiencies of 1 occur frequently in periodic settings but higher ones are rare.

    Returns:
        L: lower triangular matrix with ones on the diagonal
        U: upper triangular matrix
    """
    b = choose_backend(matrix)
    assert b.dtype(matrix).kind in (bool, int, float)
    row, col = np.indices(b.staticshape(matrix)[1:-1])
    is_lower = np.expand_dims(row > col, -1)
    is_upper = np.expand_dims(row < col, -1)
    is_diagonal = np.expand_dims(row == col, -1)
    # # --- Initialize U as the diagonal of A, then compute off-diagonal of L ---
    lower = matrix / b.expand_dims(b.get_diagonal(matrix), 1)  # Since U=diag(A), L can be computed by a simple division
    lu = matrix * is_diagonal + lower * is_lower  # combine lower + diag(A) + 0
    # --- Fixed-point iterations ---
    for sweep in range(iterations):
        diag = b.expand_dims(b.get_diagonal(lu), 1)  # should never contain 0
        sum_l_u = b.einsum('bikc,bkjc->bijc', lu * is_lower, lu * is_upper)
        l = (matrix - sum_l_u) / diag if not safe else b.divide_no_nan(matrix - sum_l_u, diag)
        lu = b.where(is_lower, l, matrix - sum_l_u)
    # --- Assemble L=lower+unit_diagonal and U. ---
    return lu * is_lower + is_diagonal, lu * ~is_lower


def incomplete_lu_coo(indices, values, shape: Tuple[int, int], iterations: int, safe: bool):
    """
    Based on *Parallel Approximate LU Factorizations for Sparse Matrices* by T.K. Huckle, https://www5.in.tum.de/persons/huckle/it_ilu.pdf.

    Every matrix in the batch must explicitly store the full diagonal.
    There should not be any zeros on the diagonal, else the LU initialization fails.

    Args:
        indices: Row & column indices of stored entries as `numpy.ndarray` of shape (batch_size, nnz, 2).
        values: Backend-compatible values tensor of shape (batch_size, nnz, channels)
        shape: Dense shape of matrix
        iterations: Number of fixed-point iterations to perform.
        safe: Avoid NaN when the rank deficiency of `matrix` is 2 or higher.
            For a rank deficiency of 1, the fixed-point algorithm will still converge without NaNs.
            If enabled, the algorithm is slightly slower.
            Rank deficiencies of 1 occur frequently in periodic settings but higher ones are rare.

    Returns:
        L: tuple `(indices, values)` where `indices` is a NumPy array and values is backend-specific
        U: tuple `(indices, values)` where `indices` is a NumPy array and values is backend-specific
    """
    assert isinstance(indices, np.ndarray), "incomplete_lu_coo indices must be a NumPy array"
    b = choose_backend(indices, values)
    assert b.dtype(values).kind in (bool, int, float)
    row, col = indices[..., 0], indices[..., 1]
    values = b.as_tensor(values)
    batch_size, nnz, channels = b.staticshape(values)
    rows, cols = shape
    assert rows == cols, "incomplete_lu_coo only implemented for square matrices"
    is_lower = np.expand_dims(row > col, -1)
    is_lower_b = b.as_tensor(is_lower)
    diagonal_indices = np.expand_dims(get_lower_diagonal_indices(row, col, shape), -1)  # indices of corresponding values that lie on the diagonal
    is_diagonal = np.expand_dims(row == col, -1)
    is_diagonal_b = b.cast(is_diagonal, b.dtype(values))
    mm_above, mm_left, mm_is_valid = strict_lu_mm_pattern_coo_batched(row, col, rows, cols)
    mm_above = b.as_tensor(np.expand_dims(mm_above, -1))
    mm_left = b.as_tensor(np.expand_dims(mm_left, -1))
    mm_is_valid = b.as_tensor(np.expand_dims(mm_is_valid, -1))
    # --- Initialize U as the diagonal of A, then compute off-diagonal of L ---
    lower = values / b.batched_gather_nd(values, diagonal_indices)  # Since U=diag(A), L can be computed by a simple division
    lu = values * is_diagonal_b + lower * b.cast(is_lower_b, b.dtype(values))  # combine lower + diag(A) + 0
    # --- Fixed-point iterations ---
    for sweep in range(iterations):
        diag = b.batched_gather_nd(lu, diagonal_indices)  # should never contain 0
        sum_l_u = b.einsum('bnkc,bnkc->bnc', b.batched_gather_nd(lu, mm_above) * mm_is_valid, b.batched_gather_nd(lu, mm_left))
        l = (values - sum_l_u) / diag if not safe else b.divide_no_nan(values - sum_l_u, diag)
        lu = b.where(is_lower_b, l, values - sum_l_u)
    # --- Assemble L=lower+unit_diagonal and U. If nnz varies along batch, keep the full sparsity pattern ---
    u_values = b.where(~is_lower_b, lu, 0)
    belongs_to_lower = (is_lower | is_diagonal)
    l_values = b.where(is_lower_b, lu, is_diagonal_b)
    u_mask_indices_b, u_mask_indices = np.where(~is_lower[..., 0])
    _, u_nnz = np.unique(u_mask_indices_b, return_counts=True)
    if np.all(u_nnz == u_nnz[0]):  # nnz for lower/upper does not vary along batch
        u_mask_indices = np.reshape(u_mask_indices, (batch_size, -1))
        u_values = b.batched_gather_nd(u_values, np.expand_dims(u_mask_indices, -1))
        u_indices = np.stack([indices[b, u_mask_indices[b], :] for b in range(batch_size)])
        _, l_mask_indices = np.where(belongs_to_lower[..., 0])
        l_mask_indices = np.reshape(l_mask_indices, (batch_size, -1))
        l_values = b.batched_gather_nd(l_values, np.expand_dims(l_mask_indices, -1))
        l_indices = np.stack([indices[b, l_mask_indices[b], :] for b in range(batch_size)])
        return (l_indices, l_values), (u_indices, u_values)
    else:  # Keep all indices since the number in lower/upper varies along the batch
        return (indices, l_values), (indices, u_values)


def strict_lu_mm_pattern_coo(row: np.ndarray, col: np.ndarray, rows, cols):
    """
    For each stored entry e at (row, col), finds the matching entries directly above and directly left of e, such that left.col == above.row.

    This is useful for multiplying a lower triangular and upper triangular matrix given the sparsity pattern but excluding the diagonals.
    The matrix multiplication then is given by
    >>> einsum('nk,nk->n', stored_lower[above_entries] * is_valid_entry, stored_upper[left_entries])

    Returns:
        above_entries: (max_num, nnz) Stored indices of matched elements above any entry.
        left_entries: (max_num, nnz) Stored indices of matched elements to the left of any entry.
        is_valid_entry: (max_num, nnz) Mask of valid indices. Invalid indices are undefined but lie inside the array to prevent index errors.
    """
    entries, = row.shape
    # --- Compress rows and cols ---
    lower_entries_by_row = compress_strict_lower_triangular_rows(row, col, rows)  # entry indices by row, -1 for non-existent entries
    upper_entries_by_col = compress_strict_lower_triangular_rows(col, row, cols)
    # --- Find above and left entries ---
    same_row_entries = lower_entries_by_row[:, row]  # (row entries, entries). Currently, contains valid values for invalid references
    left = np.where(col[same_row_entries] < col, same_row_entries, -1)  # (max_left, nnz)  all entries with col_e==col, row_e < row
    same_col_entries = upper_entries_by_col[:, col]
    above = np.where(row[same_col_entries] < row, same_col_entries, -1)  # (max_above, nnz)
    # --- for each entry, match left and above where left.col == above.row ---
    half_density = max(len(lower_entries_by_row), len(upper_entries_by_col))
    above_entries = np.zeros([entries, half_density], dtype=int)
    left_entries = np.zeros([entries, half_density], dtype=int)
    is_valid_entry = np.zeros([entries, half_density])
    k = np.zeros(entries, dtype=int)
    for r in range(len(above)):
        for c in range(len(left)):
            match = (col[left[c]] == row[above[r]]) & (above[r] != -1)
            where_match = np.where(match)
            k_where_match = k[where_match]
            above_entries[where_match, k_where_match] = above[r][where_match]
            left_entries[where_match, k_where_match] = left[c][where_match]
            is_valid_entry[where_match, k_where_match] = 1
            k += match
    return above_entries, left_entries, is_valid_entry


def compress_strict_lower_triangular_rows(row, col, rows):
    is_lower = row > col
    below_diagonal = np.where(is_lower)
    row_lower = row[below_diagonal]
    num_in_row = get_index_in_row(row_lower, col[below_diagonal])
    lower_entries_by_row = np.zeros((np.max(num_in_row)+1, rows), dtype=row.dtype) - 1
    lower_entries_by_row[num_in_row, row_lower] = below_diagonal
    return lower_entries_by_row


def strict_lu_mm_pattern_coo_batched(row, col, rows, cols):
    results = [strict_lu_mm_pattern_coo(row[b], col[b], rows, cols) for b in range(row.shape[0])]
    result = [np.stack(v) for v in zip(*results)]
    return result


def get_index_in_row(row: np.ndarray, col: np.ndarray):
    """ How many entries are to the left of a given entry but in the same row, i.e. the how manieth index this is per row. """
    perm = np.argsort(col)
    compressed_col_index = cumcount(row[perm])[inv_perm(perm)]
    return compressed_col_index


def inv_perm(perm):
    """ Returns the permutation necessary to undo a sort given the argsort array. """
    u = np.empty(perm.size, dtype=np.int64)
    u[perm] = np.arange(perm.size)
    return u


def cumcount(a):
    """ Based on https://stackoverflow.com/questions/40602269/how-to-use-numpy-to-get-the-cumulative-count-by-unique-values-in-linear-time """
    def dfill(a):
        """ Returns the positions where the array changes and repeats that index position until the next change. """
        b = np.concatenate([[0], np.where(a[:-1] != a[1:])[0] + 1, [a.size]])
        return np.arange(a.size)[b[:-1]].repeat(np.diff(b))
    perm = a.argsort(kind='mergesort')
    inv = inv_perm(perm)
    return (np.arange(a.size) - dfill(a[perm]))[inv]


def cumcount2(l):  # slightly slower than cumcount
    a = np.unique(l, return_counts=True)[1]
    idx = a.cumsum()
    id_arr = np.ones(idx[-1], dtype=int)
    id_arr[0] = 0
    id_arr[idx[:-1]] = -a[:-1] + 1
    rng = id_arr.cumsum()
    return rng[inv_perm(np.argsort(l))]


def get_transposed_indices(row, col, shape):
    linear = np.ravel_multi_index((row, col), shape)
    linear_transposed = np.ravel_multi_index((col, row), shape)
    has_transpose = np.stack([np.isin(linear[b], linear_transposed[b]) for b in range(row.shape[0])])
    perm = np.argsort(linear)
    transposed = np.stack([np.searchsorted(linear[b], linear_transposed[b], sorter=perm[b]) for b in range(row.shape[0])])
    transposed = np.minimum(transposed, len(row) - 1)
    return has_transpose, transposed


def get_lower_diagonal_indices(row, col, shape):
    linear = np.ravel_multi_index((row, col), shape)
    j = np.minimum(row, col)
    diagonal_indices = np.ravel_multi_index((j, j), shape)
    perm = np.argsort(linear)
    result = [perm[b, np.searchsorted(linear[b], diagonal_indices[b], sorter=perm[b])] for b in range(row.shape[0])]
    assert np.all([np.isin(diagonal_indices[b], linear[b]) for b in range(row.shape[0])]), "All diagonal elements must be present in sparse matrix."
    return np.stack(result)


def parallelize_dense_triangular_solve(b: Backend, matrix, lower_triangular=True):
    """

    Args:
        b:
        matrix: lower-triangular matrix

    Returns:

    """
    rows, cols = b.staticshape(matrix)
    # batch_size, rows, cols, channels = b.staticshape(matrix)
    xs = {}
    for row in range(rows):
        x = b.zeros((cols,))
        x[row] = 1
        for j in range(row):
            x -= xs[j] * matrix[row, j]
        if not lower_triangular:
            x /= matrix[row, row]
        xs[row] = x
    print(xs)


@dataclass
class ExplicitClusterSolve(Preconditioner):
    inv_coarse_matrix: TensorType
    clusters: TensorType
    cluster_count: int
    cluster_size: TensorType

    def apply(self, vec):
        b = choose_backend(vec, self.inv_coarse_matrix)
        non_batch = b.ndims(vec) == 1
        # --- Down-project / sum vec ---
        coarse_vec = b.batched_bincount(self.clusters, weights=vec[None, :] if non_batch else vec, bins=self.cluster_count)
        delta = vec - b.batched_gather_1d(coarse_vec / self.cluster_size, self.clusters)  # to make the preconditioner full-rank
        # --- direct solve ---
        coarse_solution = b.linear(self.inv_coarse_matrix, coarse_vec)
        fine_solution = b.batched_gather_1d(coarse_solution, self.clusters)  # np.set_printoptions(linewidth=np.inf); print(b.numpy(fine_solution)[0])
        result = fine_solution + delta
        return result[0] if non_batch else result

    def apply_transposed(self, vec):
        raise NotImplementedError

    def apply_inv_l(self, vec):
        return vec

    def apply_inv_u(self, vec):
        return self.apply(vec)

    def __repr__(self):
        return f"cluster ({self.cluster_count})"


def coarse_explicit_preconditioner_coo(bt: Backend, indices: TensorType, values: TensorType, shape: Tuple[int, int], clusters: TensorType, cluster_count: int) -> ExplicitClusterSolve:
    """
    Args:
        b: Target backend that performs the linear solve.
        indices: (batch_size, nnz, 2)
        values: (batch_size, nnz,)
        shape: Sparse matrix shape, (rows, cols)
        clusters: cluster index by element as (batch_size, rows/cols,)
    """
    b0 = choose_backend(indices, values, clusters)
    from . import NUMPY  # we can use NumPy inversion if either b or b0 is NumPy
    if bt is NUMPY:
        values = b0.numpy(values)  # convert(values, b)
        b = NUMPY
    else:
        b = b0
    batch_size, nnz, channels = b0.staticshape(values)
    row, col = indices[..., 0], indices[..., 1]
    assert b0.staticshape(indices)[0] == 1, f"Batched coo indices not supported"
    cluster_size = b.bincount(clusters[0, :], None, cluster_count)
    cluster_row = b.batched_gather_1d(clusters, row)
    cluster_col = b.batched_gather_1d(clusters, col)
    # --- compute coarse matrix ---
    coarse_matrix = b.zeros((batch_size, cluster_count, cluster_count, 1), b.dtype(values))
    coarse_indices = b.stack([cluster_row, cluster_col], -1)
    coarse_matrix = b.scatter(coarse_matrix, coarse_indices, values, mode='add')[..., 0]  # coarse_matrix /= cluster_size
    # --- invert matrix ---
    PHI_LOGGER.info(f"Explicit-cluster: inverting matrix of size {b.staticshape(coarse_matrix)} using NumPy")
    t = time.perf_counter()
    inv_matrix = numpy.linalg.inv(b.numpy(coarse_matrix)[0])
    PHI_LOGGER.info(f"Explicit-cluster: Matrix inverted. ({time.perf_counter() - t} seconds)")
    inv_matrix = b.as_tensor(inv_matrix)
    cluster_size_f = b.cast(cluster_size, b.dtype(values))
    return ExplicitClusterSolve(convert(inv_matrix, bt), convert(clusters, bt), cluster_count, convert(cluster_size_f, bt))


def cluster_coo(indices: np.ndarray, shape: Tuple[int, int], cluster_count: int):
    rows, cols = shape
    b = choose_backend(indices)
    batch_size, nnz, _ = b.staticshape(indices)
    avg_cluster_size = rows / cluster_count
    avg_entries_per_element = nnz / rows
    d = (avg_entries_per_element - 1) / 2  # proxy for spatial dimension
    PHI_LOGGER.info(f"Clustering matrix {shape} with {nnz} elements based on connections...")
    for b in range(batch_size):
        mask = coo_matrix((numpy.ones(nnz), indices[b].T))
        connection_counts = np.asarray(np.sum(mask, 1))[:, 0]
        start = np.where(connection_counts == connection_counts.min())[0]
        if len(start) > nnz / 2:
            start = [0]
        clusters = np.zeros((rows,), dtype=np.int32) - 1
        clusters[start] = np.arange(len(start))
        while np.any(clusters < 0):
            cluster_sizes = np.bincount(clusters + 1, minlength=len(clusters) + 1)[1:]
            """
            for each element:
                check whether it neighbors a non-full cluster
                if it connects to multiple elements (2 for d==2, 4 in 4D) of that cluster it gets priority to be added 
            """

            candidates_by_cluster = []
            raise NotImplementedError
    PHI_LOGGER.info(f"Clustering successful.")
    return ...
