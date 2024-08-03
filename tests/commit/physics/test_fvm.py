from unittest import TestCase

from phi import math
from phi.field import Field
from phi.geom import mesh_from_numpy
from phi.physics import diffuse, advect
from phiml.math import spatial, vec, tensor
from phiml.math.extrapolation import ZERO_GRADIENT


class TestSPH(TestCase):

    def test_matrix_adv_diff(self):
        points = [(0, 0), (0, 1), (1, 1), (1, 0)]
        mesh = mesh_from_numpy(points, [(0, 1, 2), (0, 2, 3)], {'x': [(1, 2), (3, 0)], 'y': [(0, 1), (2, 3)]})
        def momentum_eq(u, u_prev, dt, diffusivity=0.01):
            diffusion_term = dt * diffuse.differential(u, diffusivity, correct_skew=False)
            advection_term = dt * advect.differential(u, u_prev, order=1)
            return u + advection_term + diffusion_term
        velocity = Field(mesh, tensor(vec(x=1, y=0)), {'x': vec(x=.1, y=0), 'y': ZERO_GRADIENT})
        A, b = math.matrix_from_function(momentum_eq, velocity, velocity, 0.01)
        r_lin = A @ velocity.values + b
        r_call = momentum_eq(velocity, velocity, 0.01)
        math.assert_close(r_lin, r_call.values)

