from unittest import TestCase

from phi import math, torch
from phi.field import StaggeredGrid, CenteredGrid
from phi.geom import Box
from phi import field
from phi.physics import Domain


class TestFieldMath(TestCase):

    def test_gradient(self):
        domain = Domain(x=4, y=3)
        phi = domain.grid() * (1, 2)
        grad = field.gradient(phi, stack_dim='gradient')
        self.assertEqual(('spatial', 'spatial', 'channel', 'channel'), grad.shape.types)

    def test_divergence_centered(self):
        v = field.CenteredGrid(math.ones(x=3, y=3), Box[0:1, 0:1], math.extrapolation.ZERO) * (1, 0)  # flow to the right
        div = field.divergence(v).values
        math.assert_close(div.y[0], (1.5, 0, -1.5))

    def test_trace_function(self):
        def f(x: StaggeredGrid, y: CenteredGrid):
            return x + (y >> x)

        ft = field.trace_function(f)
        domain = Domain(x=4, y=3)
        x = domain.staggered_grid(1)
        y = domain.vector_grid(1)

        res_f = f(x, y)
        res_ft = ft(x, y)
        self.assertEqual(res_f.shape, res_ft.shape)
        field.assert_close(res_f, res_ft)

    def test_gradient_function(self):
        def f(x: StaggeredGrid, y: CenteredGrid):
            pred = x + (y >> x)
            loss = field.l2_loss(pred)
            return loss

        domain = Domain(x=4, y=3)
        x = domain.staggered_grid(1)
        y = domain.vector_grid(1)

        with torch.TORCH_BACKEND:
            dx, = field.gradient_function(f)(x, y)
            self.assertIsInstance(dx, StaggeredGrid)
            loss, dx, dy = field.gradient_function(f, (0, 1), get_output=True)(x, y)
            self.assertIsInstance(loss, math.Tensor)
            self.assertIsInstance(dx, StaggeredGrid)
            self.assertIsInstance(dy, CenteredGrid)
