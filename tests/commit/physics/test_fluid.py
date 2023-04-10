from unittest import TestCase

import phi
from phi import math, field
from phi.geom import Box, Sphere
from phi.field import StaggeredGrid, CenteredGrid, divergence, Noise
from phi.math import batch
from phi.math.backend import Backend
from phi.math.extrapolation import BOUNDARY, ZERO, PERIODIC, combine_sides
from phi.physics import fluid


BACKENDS = phi.detect_backends()


class FluidTest(TestCase):

    def _test_make_incompressible(self, grid_type: type, extrapolation: math.Extrapolation, **batch_dims):
        result = None
        for i, backend in enumerate(BACKENDS):
            with backend:
                smoke = CenteredGrid(Sphere(x=math.random_uniform(batch(**batch_dims)) * 100, y=10, radius=5), extrapolation, x=16, y=20, bounds=Box['x,y', 0:100, 0:100])
                velocity = grid_type(0, extrapolation, x=16, y=20, bounds=Box['x,y', 0:100, 0:100])
                for _ in range(2):
                    velocity += smoke * (0, 0.1) @ velocity
                    velocity, _ = fluid.make_incompressible(velocity)
                math.assert_close(0, divergence(velocity).values, abs_tolerance=2e-5)
                if result is None:
                    result = velocity
                else:
                    field.assert_close(result, abs_tolerance=1e-5, msg=f"Simulation with {backend} does not match {BACKENDS[:i]}")

    def test_make_incompressible_centered(self):
        self._test_make_incompressible(CenteredGrid, ZERO)
        self._test_make_incompressible(CenteredGrid, BOUNDARY, batch3=3, batch2=2)

    def test_make_incompressible_staggered_closed(self):
        self._test_make_incompressible(StaggeredGrid, ZERO)
        self._test_make_incompressible(StaggeredGrid, ZERO, batch3=3, batch2=2)

    def test_make_incompressible_staggered_open(self):
        self._test_make_incompressible(StaggeredGrid, BOUNDARY)
        self._test_make_incompressible(StaggeredGrid, BOUNDARY, batch3=3, batch2=2)

    def test_make_incompressible_staggered_periodic(self):
        self._test_make_incompressible(StaggeredGrid, PERIODIC)
        self._test_make_incompressible(StaggeredGrid, PERIODIC, batch3=3, batch2=2)
        
    def test_make_incompressible_staggered(self):
        ext = combine_sides(x=BOUNDARY, y=(ZERO, BOUNDARY))
        self._test_make_incompressible(StaggeredGrid, ext)
        self._test_make_incompressible(StaggeredGrid, ext, batch3=3, batch2=2)

    def test_make_incompressible_gradients_equal(self):
        def sim(velocity):
            velocity, _ = fluid.make_incompressible(velocity)
            loss = field.l2_loss(velocity)
            assert math.isfinite(loss).all
            return loss

        sim_grad = math.jacobian(sim, get_output=False)

        velocity0 = StaggeredGrid(Noise(), x=16, y=16, bounds=Box['x,y', 0:100, 0:100])
        grads = []
        for backend in BACKENDS:
            if backend.supports(Backend.jacobian):
                with backend:
                    velocity = velocity0.with_values(math.tensor(velocity0.values))
                    grad, = sim_grad(velocity)
                    assert math.isfinite(grad.values).all
                    grads.append(grad)
        math.assert_close(*grads, abs_tolerance=1e-5)
