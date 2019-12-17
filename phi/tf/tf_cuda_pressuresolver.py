import logging
import os
import numpy as np
import tensorflow as tf
from numbers import Number

from phi import math
from phi.physics.pressuresolver.solver_api import PressureSolver


if tf.__version__[0] == '2':
    logging.info('Adjusting for tensorflow 2.0')
    tf = tf.compat.v1
    tf.disable_eager_execution()


# Load Custom Ops
current_dir = os.path.dirname(os.path.realpath(__file__))
kernel_path = os.path.join(current_dir, 'cuda/build/pressure_solve_op.so')
assert os.path.isfile(kernel_path), 'CUDA binaries not found at %s. Run "python setup.py cuda" to compile them' % kernel_path
pressure_op = tf.load_op_library(kernel_path)


class CUDASolver(PressureSolver):

    def __init__(self, accuracy=1e-5, gradient_accuracy='same',
                 max_iterations=2000, max_gradient_iterations='same'):
        PressureSolver.__init__(self, 'CUDA Conjugate Gradient', supported_devices=('GPU',),
                                supports_loop_counter=True, supports_guess=False, supports_continuous_masks=False)
        assert isinstance(accuracy, Number), 'invalid accuracy: %s' % accuracy
        assert gradient_accuracy == 'same' or isinstance(gradient_accuracy, Number), 'invalid gradient_accuracy: %s' % gradient_accuracy
        assert max_gradient_iterations in ['same', 'mirror'] or isinstance(max_gradient_iterations, Number), 'invalid max_gradient_iterations: %s' % max_gradient_iterations
        self.accuracy = accuracy
        self.gradient_accuracy = accuracy if gradient_accuracy == 'same' else gradient_accuracy
        self.max_iterations = max_iterations
        if max_gradient_iterations == 'same':
            self.max_gradient_iterations = max_iterations
        elif max_gradient_iterations == 'mirror':
            self.max_gradient_iterations = 'mirror'
        else:
            self.max_gradient_iterations = max_gradient_iterations

    def solve(self, divergence, domain, pressure_guess):
        # pressure_guess: not used in this implementation, Kernel takes the last pressure value for initial_guess
        active, accessible = domain.active_tensor(extend=1), domain.accessible_tensor(extend=1)

        def pressure_gradient(op, grad):
            return cuda_solve_forward(grad, active, accessible, self.gradient_accuracy, max_gradient_iterations)[0]

        pressure, iteration = math.with_custom_gradient(
            cuda_solve_forward,
            [divergence, active, accessible, self.accuracy, self.max_iterations],
            pressure_gradient,
            input_index=0, output_index=0, name_base='cuda_pressure_solve'
        )
        max_gradient_iterations = iteration if self.max_gradient_iterations == 'mirror' else self.max_gradient_iterations
        return pressure, iteration


def cuda_solve_forward(divergence, active_mask, fluid_mask, accuracy, max_iterations):
    # Setup
    dimensions = divergence.get_shape()[1:-1]
    dimensions = dimensions[::-1]  # the custom op needs it in the x,y,z order
    dim_array = np.array(dimensions)
    dim_product = np.prod(dimensions)
    mask_dimensions = dim_array + 2
    laplace_matrix = tf.zeros(dim_product * (len(dimensions) * 2 + 1), dtype=tf.int8)
    # Helper variables for CG, make sure new memory is allocated for each variable.
    one_vector = tf.ones(dim_product, dtype=tf.float32)
    p = tf.zeros_like(divergence, dtype=tf.float32) + 1
    z = tf.zeros_like(divergence, dtype=tf.float32) + 2
    r = tf.zeros_like(divergence, dtype=tf.float32) + 3
    pressure = tf.zeros_like(divergence, dtype=tf.float32) + 4
    # Solve
    pressure, iteration = pressure_op.pressure_solve(
        dimensions, mask_dimensions, active_mask, fluid_mask, laplace_matrix,
        divergence, p, r, z, pressure, one_vector, dim_product, accuracy, max_iterations
    )
    return pressure, iteration
