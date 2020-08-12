from collections import namedtuple

from phi.backend.dynamic_backend import DYNAMIC_BACKEND as math


SolveResult = namedtuple('SolveResult', ['iterations', 'x', 'residual'])


def broyden(function, x0, inv_J0, accuracy=1e-5, max_iterations=1000, back_prop=False):
    """
    Broyden's method for finding a root of the given function.
    Given a function `f: R^n -> R^n`, it finds an x such that `f(x)=0` within the specified `accuracy`.

    Boryden's method does not require explicit computations of the Jacobian except for the initial inverse `inv_J0`.

    :param function: Differentiable black-box function mapping from tensors like x0 to tensors like y
    :param x0: initial guess for x
    :param inv_J0: Approximation of the inverse Jacobian matrix of f at x0. The closer this is to the true matrix, the fewer iterations will be required.
    :param max_iterations: (optional) maximum number of CG iterations to perform
    :param back_prop: Whether to enable auto-differentiation. This induces a memory cost scaling with the number of iterations. Otherwise, the memory cost is constant.
    :param accuracy: (optional) the algorithm terminates once |f(x)| ≤ accuracy for every entry. If None, the algorithm runs until `max_iterations` is reached.
    :return: list of SolveResults with [0] being the forward solve result, [1] backward solve result (will be added once backward pass is computed)
    :rtype: SolveResult
    """
    x0 = math.to_float(x0)
    y0 = function(x0)
    inv_J0 = math.to_float(inv_J0)

    def broyden_loop(x, y, inv_J, iterations):
        # --- Adjust our guess for x ---
        dx = - math.einsum('bij,bj->bi', inv_J, y)  # - J^-1 * y
        next_x = x + dx
        next_y = function(next_x)
        df = next_y - y
        dy_back_projected = math.einsum('bij,bj->bi', inv_J, df)
        # --- Approximate next inverted Jacobian ---
        numerator = math.einsum('bi,bj,bjk->bik', dx - dy_back_projected, dx, inv_J)  # (dx - J^-1 * df) * dx^T * J^-1
        denominator = math.einsum('bi,bi->b', dx, dy_back_projected)  # dx^T * J^-1 * df
        next_inv_J = inv_J + numerator / denominator
        return [next_x, next_y, next_inv_J, iterations + 1]

    x_, y_, _, iterations = math.while_loop(_max_residual_condition(1, accuracy), broyden_loop, [x0, y0, inv_J0, 0], back_prop=back_prop, name='Broyden', maximum_iterations=max_iterations)
    return SolveResult(iterations, x_, y_)


def conjugate_gradient(function, y, x0, accuracy=1e-5, max_iterations=1000, back_prop=False):
    """
    Solve the linear system of equations `A·x=y`  using the conjugate gradient (CG) algorithm.
    A, x and y can have arbitrary matching shapes, i.e. this method can be used to solve vector and matrix equations.

    Since representing the matrix A in memory might not be feasible, a linear function of x can be specified that computes `function(x) = Ax`.

    The implementation is based on https://nvlpubs.nist.gov/nistpubs/jres/049/jresv49n6p409_A1b.pdf

    :param y: Desired output of `f(x)`
    :param function: linear function of x that returns A·x
    :param x0: initial guess for the value of x
    :param accuracy: (optional) the algorithm terminates once |f(x)-y| ≤ accuracy for every entry. If None, the algorithm runs until `max_iterations` is reached.
    :param max_iterations: (optional) maximum number of CG iterations to perform
    :param back_prop: Whether to enable auto-differentiation. This induces a memory cost scaling with the number of iterations. Otherwise, the memory cost is constant.
    :return: Pair containing the result for x and the number of iterations performed
    """
    y = math.to_float(y)
    x0 = math.to_float(x0)
    dx0 = residual0 = y - function(x0)
    dy0 = function(dx0)
    non_batch_dims = tuple(range(1, len(y.shape)))

    def cg_loop(x, dx, dy, residual, iterations):
        dx_dy = math.sum(dx * dy, axis=non_batch_dims, keepdims=True)
        step_size = math.divide_no_nan(math.sum(dx * residual, axis=non_batch_dims, keepdims=True), dx_dy)
        x += step_size * dx
        residual -= step_size * dy
        dx = residual - math.divide_no_nan(math.sum(residual * dy, axis=non_batch_dims, keepdims=True) * dx, dx_dy)
        dy = function(dx)
        return [x, dx, dy, residual, iterations + 1]

    x_, _, _, residual_, iterations_ = math.while_loop(_max_residual_condition(3, accuracy), cg_loop, [x0, dx0, dy0, residual0, 0], back_prop=back_prop, name="ConjGrad", maximum_iterations=max_iterations)
    return SolveResult(iterations_, x_, residual_)


def _max_residual_condition(residual_index, accuracy):
    """continue if the maximum deviation from zero is bigger than desired accuracy"""
    if accuracy is None:
        return lambda *args: True
    else:
        return lambda *args: math.max(math.abs(args[residual_index])) > accuracy
