from unittest import TestCase

import numpy as np

from phi.geom._mesh_builder import MeshBuilder
from phiml.math import math, spatial, vec, wrap


def _point_in_polygon(point, polygon):
    x, y = point
    inside = False
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-30) + x1):
            inside = not inside
    return inside


class TestMeshBuilder(TestCase):

    def test_triangulate_concave_loop(self):
        points_np = np.array([
            (0.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (4.0, 4.0, 0.0),
            (3.0, 4.0, 0.0),
            (3.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 4.0, 0.0),
            (0.0, 4.0, 0.0),
        ], dtype=np.float32)
        points = wrap(points_np, 'poly:s,(x,y,z)')

        builder = MeshBuilder(2)
        builder.add_vertices('poly', points)
        builder.triangulate(wrap([0, 1, 2, 3, 4, 5, 6, 7, 0], spatial('loop')))
        mesh = builder.build_mesh()

        math.assert_close(6, mesh.cell_count)
        math.assert_close(10.0, mesh.volume.sum)

        centers = mesh.center
        polygon = points_np[:, :2]
        for center in centers.elements:
            self.assertTrue(_point_in_polygon(center[:2], polygon), msg=f"Triangle center {center} should lie inside the polygon")

