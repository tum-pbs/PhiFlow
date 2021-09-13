from unittest import TestCase

import phi
from phi import math, field
from phi.geom import Box, Sphere
from phi.field import StaggeredGrid, CenteredGrid, divergence, Noise
from phi.math import batch
from phi.math.backend import Backend
from phi.physics import fluid


BACKENDS = phi.detect_backends()


class FluidTest(TestCase):

    def _test_make_incompressible(self, grid_type: type, extrapolation: math.Extrapolation, **batch_dims):
        result = None
        for i, backend in enumerate(BACKENDS):
            with backend:
                smoke = CenteredGrid(Sphere(center=(math.random_uniform(batch(**batch_dims)) * 100, 10), radius=5), extrapolation, x=16, y=20, bounds=Box[0:100, 0:100])
                velocity = grid_type(0, extrapolation, x=16, y=20, bounds=Box[0:100, 0:100])
                for _ in range(2):
                    velocity += smoke * (0, 0.1) @ velocity
                    velocity, _ = fluid.make_incompressible(velocity)
                math.assert_close(divergence(velocity).values, 0, abs_tolerance=2e-5)
                if result is None:
                    result = velocity
                else:
                    field.assert_close(result, abs_tolerance=1e-5, msg=f"Simulation with {backend} does not match {BACKENDS[:i]}")

    def test_make_incompressible_centered(self):
        self._test_make_incompressible(CenteredGrid, math.extrapolation.ZERO)
        self._test_make_incompressible(CenteredGrid, math.extrapolation.BOUNDARY, batch3=3, batch2=2)

    def test_make_incompressible_staggered_closed(self):
        self._test_make_incompressible(StaggeredGrid, math.extrapolation.ZERO)
        self._test_make_incompressible(StaggeredGrid, math.extrapolation.ZERO, batch3=3, batch2=2)

    def test_make_incompressible_staggered_open(self):
        self._test_make_incompressible(StaggeredGrid, math.extrapolation.BOUNDARY)
        self._test_make_incompressible(StaggeredGrid, math.extrapolation.BOUNDARY, batch3=3, batch2=2)

    def test_make_incompressible_staggered_periodic(self):
        self._test_make_incompressible(StaggeredGrid, math.extrapolation.PERIODIC)
        self._test_make_incompressible(StaggeredGrid, math.extrapolation.PERIODIC, batch3=3, batch2=2)

    def test_make_incompressible_gradients_equal_tf_torch(self):
        velocity0 = StaggeredGrid(Noise(), math.extrapolation.ZERO, x=16, y=16, bounds=Box[0:100, 0:100])
        grads = []
        for backend in BACKENDS:
            if backend.supports(Backend.record_gradients):
                with backend:
                    velocity = param = velocity0.with_values(math.tensor(velocity0.values))
                    with math.record_gradients(param.values):
                        velocity, _ = fluid.make_incompressible(velocity)
                        loss = field.l2_loss(velocity)
                        assert math.isfinite(loss).all
                        grad = math.gradients(loss, param.values)
                        assert math.isfinite(grad).all
                        grads.append(grad)
        math.assert_close(*grads, abs_tolerance=1e-5)
