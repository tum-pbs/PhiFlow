from unittest import TestCase

from phi import math
from phi.geom import Box, Sphere
from phi.field import StaggeredGrid, CenteredGrid
from phi.physics import Domain, CLOSED, fluid
from phi.tf import TF_BACKEND
from phi.torch import TORCH_BACKEND

import tensorflow as tf
if tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)  # prevent Blas GEMM launch failed


class FluidTest(TestCase):

    # Backend-independent abstract tests

    def _test_make_incompressible(self, grid_type):
        DOMAIN = Domain(x=16, y=16, boundaries=CLOSED, bounds=Box[0:100, 0:100])
        smoke = DOMAIN.grid(Sphere(center=(50, 10), radius=5))
        velocity = DOMAIN.vector_grid(0, grid_type)
        for _ in range(2):
            velocity += smoke * (0, 0.1) >> velocity
            velocity, pressure, iterations, divergence = fluid.make_incompressible(velocity, DOMAIN)
        return velocity.values

    def _test_make_incompressible_batched(self, grid_type):
        DOMAIN = Domain(x=16, y=16, boundaries=CLOSED, bounds=Box[0:100, 0:100])
        smoke = DOMAIN.grid(Sphere(center=(math.random_uniform(batch=2) * 100, 10), radius=5))
        velocity = DOMAIN.vector_grid(0, grid_type)
        for _ in range(2):
            velocity += smoke * (0, 0.1) >> velocity
            velocity, pressure, iterations, divergence = fluid.make_incompressible(velocity, DOMAIN)
        return velocity.values

    # Backend-specific testing

    def test_make_incompressible_staggered_scipy(self):
        with math.SCIPY_BACKEND:
            self._test_make_incompressible(StaggeredGrid)
            self._test_make_incompressible_batched(StaggeredGrid)

    def test_make_incompressible_centered_scipy(self):
        with math.SCIPY_BACKEND:
            self._test_make_incompressible(CenteredGrid)
            self._test_make_incompressible_batched(CenteredGrid)

    def test_make_incompressible_staggered_tensorflow(self):
        with TF_BACKEND:
            self._test_make_incompressible(StaggeredGrid)
            self._test_make_incompressible_batched(StaggeredGrid)

    def test_make_incompressible_centered_tensorflow(self):
        with TF_BACKEND:
            self._test_make_incompressible(CenteredGrid)
            self._test_make_incompressible_batched(CenteredGrid)

    def test_make_incompressible_staggered_pytorch(self):
        with TORCH_BACKEND:
            self._test_make_incompressible(StaggeredGrid)
            self._test_make_incompressible_batched(StaggeredGrid)

    def test_make_incompressible_centered_pytorch(self):
        with TORCH_BACKEND:
            self._test_make_incompressible(CenteredGrid)
            self._test_make_incompressible_batched(CenteredGrid)

    def test_make_incompressible_np_equal_tf(self):
        with math.SCIPY_BACKEND:
            v_np = self._test_make_incompressible(StaggeredGrid)
        with TF_BACKEND:
            v_tf = self._test_make_incompressible(StaggeredGrid)
        math.assert_close(v_np, v_tf, abs_tolerance=1e-5)

    def test_make_incompressible_np_equal_torch(self):
        with math.SCIPY_BACKEND:
            v_np = self._test_make_incompressible(StaggeredGrid)
        with TORCH_BACKEND:
            v_to = self._test_make_incompressible(StaggeredGrid)
        math.assert_close(v_np, v_to, abs_tolerance=1e-5)
