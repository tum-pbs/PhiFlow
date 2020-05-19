from collections import namedtuple

from phi.backend.dynamic_backend import DYNAMIC_BACKEND as math


SolveResult = namedtuple('SolveResult', ['function', 'y', 'x0', 'iterations', 'x', 'f'])


def broyden(function, x0, inv_J0, accuracy=1e-5, max_iterations=1000, back_prop=False):
    """
    Broyden's method for finding a root of the given function.
    Given a function f: R^n -> R^n, it finds an x such that f(x) = 0 (within the specified `accuracy`).

    Boryden's method does not require explicit computations of the Jacobian except for the initial inverse `inv_J0`.

    :param function: Differentiable black-box function mapping from tensors like x0 to tensors like y
    :param x0: initial guess for x
    :param inv_J0: Inverse jacobian matrix at x0. This can be an approximation but the number of Broyden iterations increases the further this is from the true matrix.
    :return: list of SolveResults with [0] being the forward solve result, [1] backward solve result (will be added once backward pass is computed)
    :rtype: SolveResult
    """
    x = math.to_float(x0)
    f = function(x)
    inv_J = math.to_float(inv_J0)

    def loop_condition(_x, f, _inv_J, _iter):
        return math.max(math.abs(f)) > accuracy

    def loop_body(x, f, inv_J, iter):
        # --- Adjust our guess ---
        dx = - math.einsum('bij,bj->bi', inv_J, f)  # - J^-1 * f
        next_x = x + dx
        next_f = function(next_x)
        df = next_f - f
        df_back_projected = math.einsum('bij,bj->bi', inv_J, df)
        # --- Approximate next inverted Jacobian ---
        numerator = math.einsum('bi,bj,bjk->bik', dx - df_back_projected, dx, inv_J)  # (dx - J^-1 * df) * dx^T * J^-1
        denominator = math.einsum('bi,bi->b', dx, df_back_projected)  # dx^T * J^-1 * df
        next_inv_J = inv_J + numerator / denominator
        return [next_x, next_f, next_inv_J, iter + 1]

    x, f, inv_J, iter = math.while_loop(loop_condition, loop_body, [x, f, inv_J, 0], back_prop=back_prop, name='Broyden', maximum_iterations=max_iterations)
    return SolveResult(function, 0, x0, iter, x, f)


def conjugate_gradient(k, apply_A, initial_x=None, accuracy=1e-5, max_iterations=1024, back_prop=False):
    """
    Solve the linear system of equations Ax=k using the conjugate gradient (CG) algorithm.
    The implementation is based on https://nvlpubs.nist.gov/nistpubs/jres/049/jresv49n6p409_A1b.pdf

    :param k: Right-hand-side vector
    :param apply_A: function that takes x and calculates Ax
    :param initial_x: initial guess for the value of x
    :param accuracy: the algorithm terminates once |Ax-k| ≤ accuracy for every element. If None, the algorithm runs until max_iterations is reached.
    :param max_iterations: maximum number of CG iterations to perform
    :return: Pair containing the result for x and the number of iterations performed
    """
    k = math.copy(k, only_mutable=True)
    # Get momentum = k - Ax
    if initial_x is None:
        x = math.zeros_like(k)
        momentum = k
    else:
        x = math.copy(initial_x, only_mutable=True)
        momentum = k - apply_A(x)
    # Further Variables
    residual = momentum  # residual is previous momentum
    laplace_momentum = apply_A(momentum)  # = A*momentum
    loop_index = 0  # initial
    # Pack Variables for loop
    variables = [x, momentum, laplace_momentum, residual, loop_index]
    # Ensure to run until desired accuracy is achieved
    if accuracy is not None:
        def loop_condition(_1, _2, _3, residual, _i):
            """continue if the maximum deviation from zero is bigger than desired accuracy"""
            return math.max(math.abs(residual)) > accuracy
    else:
        def loop_condition(*_args):
            return True

    non_batch_dims = tuple(range(1, len(k.shape)))

    def loop_body(pressure, momentum, A_times_momentum, residual, loop_index):
        tmp = math.sum(momentum * A_times_momentum, axis=non_batch_dims, keepdims=True)  # t = sum(mAm)
        a = math.divide_no_nan(math.sum(momentum * residual, axis=non_batch_dims, keepdims=True), tmp)  # a = sum(mr)/sum(mAm)
        pressure += a * momentum  # p += am
        residual -= a * A_times_momentum  # r -= aAm
        momentum = residual - math.divide_no_nan(math.sum(residual * A_times_momentum, axis=non_batch_dims, keepdims=True) * momentum, tmp)  # m = r-sum(rAm)*m/t = r-sum(rAm)*m/sum(mAm)
        A_times_momentum = apply_A(momentum)  # Am = A*m
        return [pressure, momentum, A_times_momentum, residual, loop_index + 1]

    x, momentum, laplace_momentum, residual, loop_index = math.while_loop(loop_condition, loop_body, variables, parallel_iterations=2, back_prop=back_prop, name="ConjugateGradient", maximum_iterations=max_iterations)
    return x, loop_index
