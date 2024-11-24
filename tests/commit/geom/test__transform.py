from unittest import TestCase

from phi import math
from phi.geom import rotate, rotation_matrix, rotation_angles
from phiml.math import wrap, batch


class TestGeom(TestCase):

    def test_rotate_vector(self):
        # --- 2D ---
        vec = rotate(math.vec(x=2, y=0), math.PI / 2)
        math.assert_close(math.vec(x=0, y=2), vec, abs_tolerance=1e-5)
        math.assert_close(math.vec(x=2, y=0), rotate(vec, math.PI / 2, invert=True), abs_tolerance=1e-5)
        # --- 3D ---
        vec = rotate(math.vec(x=2, y=0, z=0), rot=math.vec(x=0, y=math.PI / 2, z=0))
        math.assert_close(math.vec(x=0, y=0, z=-2), vec, abs_tolerance=1e-5)
        math.assert_close(math.vec(x=2, y=0, z=0), rotate(vec, rot=math.vec(x=0, y=math.PI / 2, z=0), invert=True), abs_tolerance=1e-5)
        # --- None ---
        math.assert_close(math.vec(x=2, y=0), rotate(math.vec(x=2, y=0), None, invert=True))

    def test_rotation_matrix(self):
        def assert_matrices_equal(angle):
            matrix = rotation_matrix(angle)
            angle_ = rotation_angles(matrix)
            math.assert_close(matrix, rotation_matrix(angle_), abs_tolerance=1e-5)

        angle = wrap([0, -math.PI/2, math.PI/2, -math.PI, math.PI, 2*math.PI], batch('angles'))
        assert_matrices_equal(angle)
        # --- 3D axis-angle ---
        angle = wrap([0, -math.PI/2, math.PI/2, -math.PI, math.PI, 2*math.PI], batch('angles'))
        assert_matrices_equal(math.vec(x=0, y=0, z=angle))
        assert_matrices_equal(math.vec(x=0, y=angle, z=0))
        assert_matrices_equal(math.vec(x=angle, y=0, z=0))
        assert_matrices_equal(math.vec(x=angle, y=angle, z=0))
        assert_matrices_equal(math.vec(x=angle, y=angle, z=angle))
        # --- 3D Euler angle ---
        angle = wrap([0, -math.PI/2, math.PI/2, -math.PI, math.PI, 2*math.PI], batch('angles'))
        assert_matrices_equal(math.vec('angle', x=0, y=0, z=angle))
        assert_matrices_equal(math.vec('angle', x=0, y=angle, z=0))
        assert_matrices_equal(math.vec('angle', x=angle, y=0, z=0))
        assert_matrices_equal(math.vec('angle', x=angle, y=angle, z=0))
        assert_matrices_equal(math.vec('angle', x=angle, y=angle, z=angle))