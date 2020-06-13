import os
import numpy as np

from . import tf
from phi import math
from phi.physics.pressuresolver.solver_api import PoissonSolver

# --- Load Custom Ops ---
current_dir = os.path.dirname(os.path.realpath(__file__))
kernel_path = os.path.join(current_dir, 'cuda/build/pressure_solve_op.so')
if not os.path.isfile(kernel_path):
    raise ImportError('CUDA binaries not found at %s. Run "python setup.py tf_cuda" to compile them' % kernel_path)
pressure_op = tf.load_op_library(kernel_path)


class CUDASolver(PoissonSolver):

    def __init__(self, accuracy=1e-5, max_iterations=2000):
        PoissonSolver.__init__(self, 'CUDA Conjugate Gradient', supported_devices=('GPU',), supports_loop_counter=True, supports_guess=False, supports_continuous_masks=False)
        self.accuracy = accuracy
        self.max_iterations = max_iterations

    def solve(self, divergence, domain, guess, enable_backprop):
        """
        :param guess: not used in this implementation, Kernel takes the last pressure value for initial_guess
        """
        active_mask, accessible_mask = domain.active_tensor(extend=1), domain.accessible_tensor(extend=1)
        # Setup
        dimensions = math.staticshape(divergence)[1:-1]
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
            dimensions, mask_dimensions, active_mask, accessible_mask, laplace_matrix,
            divergence, p, r, z, pressure, one_vector, dim_product, self.accuracy, self.max_iterations
        )
        return pressure, iteration
