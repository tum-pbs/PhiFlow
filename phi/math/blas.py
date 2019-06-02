# coding=utf-8
from phi import math


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
    if initial_x is None:
        x = math.zeros_like(k)
        momentum = k
    else:
        x = initial_x
        momentum = k - apply_A(x)
    residual = momentum

    laplace_momentum = apply_A(momentum)
    loop_index = 0

    vars = [x, momentum, laplace_momentum, residual, loop_index]

    if accuracy is not None:
        def loop_condition(_1, _2, _3, residual, i):
            return math.max(math.abs(residual)) >= accuracy
    else:
        def loop_condition(_1, _2, _3, residual, i):
            return True

    def loop_body(pressure, momentum, A_times_momentum, residual, loop_index):
        tmp = math.sum(momentum * A_times_momentum)
        a = math.sum(momentum * residual) / tmp
        pressure += a * momentum
        residual -= a * A_times_momentum
        b = - math.sum(residual * A_times_momentum) / tmp
        momentum = residual + b * momentum
        A_times_momentum = apply_A(momentum)
        return [pressure, momentum, A_times_momentum, residual, loop_index + 1]

    x, momentum, laplace_momentum, residual, loop_index = math.while_loop(loop_condition, loop_body, vars,
                                                                              parallel_iterations=2, back_prop=back_prop,
                                                                              swap_memory=False,
                                                                              name="pressure_solve_loop",
                                                                              maximum_iterations=max_iterations)

    return x, loop_index
