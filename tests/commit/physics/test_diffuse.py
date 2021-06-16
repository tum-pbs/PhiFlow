from unittest import TestCase

from phi import math, field
from phi.field import CenteredGrid, Noise, StaggeredGrid
from phi.math import extrapolation, NotConverged, batch
from phi.physics import diffuse


class TestDiffusion(TestCase):

    def test_diffuse_centered_batched(self):
        grid = CenteredGrid(Noise(batch=2, vector=2), extrapolation.PERIODIC, x=4, y=3)
        diffuse.explicit(grid, 1, 1, substeps=10)
        diffuse.implicit(grid, 1, 1, order=2)
        diffuse.fourier(grid, 1, 1)

    def test_diffuse_staggered_batched(self):
        for diffusivity in [1, 0.5, math.wrap([1., 0.5], batch('batch'))]:
            grid = StaggeredGrid(Noise(batch(batch=2), vector=2), extrapolation.PERIODIC, x=4, y=3)
            diffuse.explicit(grid, diffusivity, 1, substeps=10)
            diffuse.implicit(grid, diffusivity, 1, order=2)
            diffuse.fourier(grid, diffusivity, 1)
            grid = StaggeredGrid(Noise(batch(batch=2), vector=2), extrapolation.ZERO, x=4, y=3)
            diffuse.explicit(grid, diffusivity, 1, substeps=10)
            # diffuse.implicit(grid, diffusivity, 1, order=2)  # not yet supported
            grid = StaggeredGrid(Noise(batch(batch=2), vector=2), extrapolation.BOUNDARY, x=4, y=3)
            diffuse.explicit(grid, diffusivity, 1, substeps=10)
            # diffuse.implicit(grid, diffusivity, 1, order=2)  # not yet supported

    def test_constant_diffusion(self):
        grid = CenteredGrid(1, extrapolation.PERIODIC, x=4, y=3)
        explicit = diffuse.explicit(grid, 1, 1, substeps=10)
        implicit = diffuse.implicit(grid, 1, 1, order=2)
        fourier = diffuse.fourier(grid, 1, 1)
        math.assert_close(grid.values, explicit.values, implicit.values, fourier.values)

    def test_equality_1d_periodic(self):
        DIFFUSIVITY = 0.5
        grid = CenteredGrid((1,) * 100 + (0,) * 100, extrapolation.PERIODIC, x=200)
        explicit = diffuse.explicit(grid, DIFFUSIVITY, 1, substeps=1000)
        implicit = diffuse.implicit(grid, DIFFUSIVITY, 1, order=10)
        fourier = diffuse.fourier(grid, DIFFUSIVITY, 1)
        field.assert_close(explicit, implicit, rel_tolerance=0, abs_tolerance=0.01)
        field.assert_close(explicit, implicit, fourier, rel_tolerance=0, abs_tolerance=0.1)
        # print(f"{explicit.values[:6]}  Explicit")
        # print(f"{implicit.values[:6]}  Implicit")
        # print(f"{fourier.values[:6]}  Fourier")
        # print()
        back_explicit = diffuse.explicit(explicit, DIFFUSIVITY, -1, substeps=1000)
        back_implicit = diffuse.implicit(implicit, DIFFUSIVITY, -1, order=10)
        back_fourier = diffuse.fourier(fourier, DIFFUSIVITY, -1)
        # print(f"{back_explicit.values[:6]}  Explicit")
        # print(f"{back_implicit.values[:6]}  Implicit")
        # print(f"{back_fourier.values[:6]}  Fourier")
        field.assert_close(grid, back_explicit, back_implicit, back_fourier, rel_tolerance=0, abs_tolerance=0.1)

    def test_consistency_implicit(self):
        DIFFUSIVITY = 0.5
        grid = CenteredGrid((1,) * 100 + (0,) * 100, extrapolation.PERIODIC, x=200)
        for extrap in (extrapolation.ZERO, extrapolation.BOUNDARY, extrapolation.PERIODIC):
            grid = grid.with_(extrapolation=extrap)
            implicit = diffuse.implicit(grid, DIFFUSIVITY, 1, order=10)
            back_implicit = diffuse.implicit(implicit, DIFFUSIVITY, -1, order=10)
            field.assert_close(grid, back_implicit, rel_tolerance=0, abs_tolerance=0.1)

    def test_implicit_stability(self):
        DIFFUSIVITY = 10
        grid = CenteredGrid((1,) * 3 + (0,) * 3, extrapolation.PERIODIC, x=6)
        try:
            implicit = diffuse.implicit(grid, DIFFUSIVITY, 1, order=10)
            print(implicit.values)
            field.assert_close(0 <= implicit <= 1.0001, True)
        except NotConverged as err:
            print(err)
            pass  # solve_linear did not converge
