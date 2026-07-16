from typing import Union, Dict
from importlib import import_module
from math import acos

import numpy as np

from phiml.math import Tensor, range_tensor, non_spatial, spatial, instance, Shape, EMPTY_SHAPE, stack, channel, expand, wrap
from ._mesh import Mesh, mesh_from_numpy


def _triangle_quality_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[float, float, float, float]:
    ab = float(np.linalg.norm(b - a))
    bc = float(np.linalg.norm(c - b))
    ca = float(np.linalg.norm(a - c))
    if min(ab, bc, ca) <= 0:
        return (-np.inf, -np.inf, -np.inf, -np.inf)
    cos_a = (ab * ab + ca * ca - bc * bc) / (2 * ab * ca)
    cos_b = (ab * ab + bc * bc - ca * ca) / (2 * ab * bc)
    cos_c = (bc * bc + ca * ca - ab * ab) / (2 * bc * ca)
    cos_a = -1.0 if cos_a < -1.0 else 1.0 if cos_a > 1.0 else cos_a
    cos_b = -1.0 if cos_b < -1.0 else 1.0 if cos_b > 1.0 else cos_b
    cos_c = -1.0 if cos_c < -1.0 else 1.0 if cos_c > 1.0 else cos_c
    angle_a = float(acos(cos_a))
    angle_b = float(acos(cos_b))
    angle_c = float(acos(cos_c))
    return float(min(angle_a, angle_b, angle_c)), float(-max(angle_a, angle_b, angle_c)), float(-(ab + bc + ca)), float(-max(ab, bc, ca))


def _polygon_area_2d(points: np.ndarray):
    return 0.5 * np.sum(points[:-1, 0] * points[1:, 1] - points[1:, 0] * points[:-1, 1]) + 0.5 * (points[-1, 0] * points[0, 1] - points[0, 0] * points[-1, 1])


def _cross2d(u: np.ndarray, v: np.ndarray):
    return float(u[0] * v[1] - u[1] * v[0])


def _point_in_triangle_2d(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, eps=1e-12):
    return (_cross2d(b - a, point - a) > eps) and (_cross2d(c - b, point - b) > eps) and (_cross2d(a - c, point - c) > eps)


def _triangulate_polygon_fast(points_2d: np.ndarray, vertex_ids: np.ndarray):
    try:
        Polygon = import_module("shapely.geometry").Polygon
        import trimesh
    except Exception:
        return None
    try:
        polygon = Polygon(points_2d)
        if not polygon.is_valid or polygon.area <= 0:
            return None
        vertices, faces = trimesh.creation.triangulate_polygon(polygon)
    except Exception:
        return None
    keys = {tuple(np.round(p, 12)): int(i) for i, p in zip(vertex_ids, points_2d)}
    try:
        mapped = np.array([[keys[tuple(np.round(v, 12))] for v in tri] for tri in vertices[faces]], dtype=np.int32)
    except Exception:
        return None
    return mapped


def _triangulate_polygon_fallback(points_2d: np.ndarray, vertex_ids: np.ndarray):
    points_2d = np.asarray(points_2d, dtype=float)
    vertex_ids = np.asarray(vertex_ids, dtype=np.int32)
    if vertex_ids.shape[0] == 3:
        return vertex_ids[None]
    if vertex_ids.shape[0] < 3:
        raise ValueError(f"Cannot triangulate loop with fewer than 3 vertices but got {vertex_ids.shape[0]}")
    signed_area = _polygon_area_2d(points_2d)
    if abs(signed_area) < 1e-14:
        raise ValueError(f"Cannot triangulate degenerate loop with near-zero area {signed_area}")
    reverse_output = signed_area < 0
    if reverse_output:
        points_2d = points_2d[::-1]
        vertex_ids = vertex_ids[::-1]
    active = list(range(vertex_ids.shape[0]))
    triangles = []
    max_iter = vertex_ids.shape[0] ** 2 + 1
    while len(active) > 3 and max_iter > 0:
        max_iter -= 1
        best = None
        best_score = None
        m = len(active)
        for pos, curr in enumerate(active):
            prev_idx = active[pos - 1]
            next_idx = active[(pos + 1) % m]
            a, b, c = points_2d[prev_idx], points_2d[curr], points_2d[next_idx]
            if _cross2d(b - a, c - a) <= 1e-12:
                continue
            if any(_point_in_triangle_2d(points_2d[other], a, b, c) for other in active if other not in (prev_idx, curr, next_idx)):
                continue
            score = _triangle_quality_2d(a, b, c)
            if best is None or score > best_score:
                best = (prev_idx, curr, next_idx)
                best_score = score
        if best is None:
            # Numerical fallback: clip the first available ear-like triangle.
            curr = active[0]
            best = (active[-1], curr, active[1])
        triangles.append(vertex_ids[list(best)])
        active.remove(best[1])
    triangles.append(vertex_ids[active])
    triangles = np.asarray(triangles, dtype=np.int32)
    if reverse_output:
        triangles = triangles[:, [0, 2, 1]]
    return triangles


def _triangulate_loop(points: np.ndarray, vertex_ids: np.ndarray, flip=False):
    points = np.asarray(points, dtype=float)
    vertex_ids = np.asarray(vertex_ids, dtype=np.int32)
    if vertex_ids.shape[0] > 1 and vertex_ids[0] == vertex_ids[-1]:
        vertex_ids = vertex_ids[:-1]
        points = points[:-1]
    if points.shape[-1] > 2:
        centered = points - np.mean(points, axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        points_2d = centered @ vh[:2].T
    else:
        points_2d = points[..., :2]
    triangles = _triangulate_polygon_fast(points_2d, vertex_ids)
    if triangles is None:
        triangles = _triangulate_polygon_fallback(points_2d, vertex_ids)
    if flip:
        triangles = triangles[:, [0, 2, 1]]
    return triangles


class MeshBuilder:
    def __init__(self, element_rank: int, batch_dims: Shape = None, source_face_shape: Shape = None):
        self.element_rank = element_rank
        self.batch_dims = batch_dims or EMPTY_SHAPE
        self.axes = None
        self.v_buffer = np.empty((self.batch_dims.volume, 0, 3))
        self.v_positions: Dict[str, Tensor] = {}
        self.v_indices: Dict[str, Tensor] = {}
        self.elements = []
        self.source_face_shape = source_face_shape
        self.source_idx = [] if source_face_shape is not None else None

    def build_mesh(self, element_dim=instance('elements')) -> Mesh:
        meshes = []
        for i in range(self.batch_dims.volume):
            elements = [e[i] for e in self.elements]
            mesh = mesh_from_numpy(self.v_buffer[i], elements, {}, self.element_rank, axes=self.axes, cell_dim=element_dim)
            meshes.append(mesh)
        return stack(meshes, self.batch_dims)

    def build_displaced_mesh(self, distance: Union[float, Tensor], element_dim=instance('elements')) -> 'MeshBuilder':
        distance = wrap(distance)
        meshes = []
        for bi, b in enumerate(self.batch_dims.meshgrid()):
            distance_np = expand(distance[b], self.source_face_shape).numpy([*self.source_face_shape])
            new_vertices = []
            new_elements = []
            for element, idx in zip(self.elements, self.source_idx):
                v = self.v_buffer[bi, element[bi, :], :]
                normal = np.cross(v[1] - v[0], v[2] - v[0])
                offset = normal / np.linalg.norm(normal) * distance_np[tuple(idx[bi])]
                new_elements.append(np.arange(len(v)) + len(new_vertices))
                new_vertices.extend(v + offset)
            mesh = mesh_from_numpy(new_vertices, new_elements, {}, self.element_rank, axes=self.axes, cell_dim=element_dim)
            meshes.append(mesh)
        return stack(meshes, self.batch_dims)

    def add_vertices(self, name: str, points: Tensor):
        """

        Args:
            name: Name of the vertex group, can be used in `MeshBuilder.vertex_indices()` and `MeshBuilder.vertices()` to retrieve the vertices later.
            points: Vertex positions, shape `(..., vector:c)` where any dimensions can be given in addition to vector.

        Returns:
            Index tensor of the added vertices, shape `(...)`, i.e. same as `points.shape - 'vector'`.
        """
        if self.axes is None:
            self.axes = points.vector.item_names
        s = points.shape - 'vector'
        idx = self.v_buffer.shape[1] + range_tensor(s - self.batch_dims)
        self.v_indices[name] = idx
        self.v_positions[name] = points
        self.v_buffer = np.concatenate([self.v_buffer, points.numpy([self.batch_dims, s-self.batch_dims, 'vector'])], -2)
        return idx

    def vertices(self, name: str) -> Tensor:
        return self.v_positions[name]

    def vertex_indices(self, name: str) -> Tensor:
        return self.v_indices[name]

    def new_quads(self, name: str, points: Tensor, source_idx: Tensor, /, flip: Union[Tensor, bool] = False):
        indices = self.add_vertices(name, points)
        self.add_quads(indices, source_idx, flip=flip)
        return indices

    def add_quads(self, indices2d: Tensor | list[list[Tensor]], source_idx: Tensor = None, /, mask: Tensor = None, flip: Union[Tensor, bool] = False):
        """
        Add quads to the mesh, connecting previously added vertices.

        Args:
            indices2d: 2D tensor of vertex indices, shape `(..., u:s, v:s)`.
                Use `MeshBuilder.vertex_indices()` or the output of `add_vertices()` to get indices of existing vertices.
            source_idx: Meta-information about the added quads, can have fewer dims than `indices2d`.
            mask: Optional mask to specify a subset of quads to be added (only at True). Must have one fewer entries along spatial dims of `indices2d`.
            flip: Whether to flip the quad orientation, i.e. reverse the order in which the vertices are listed per quad.
                Can have fewer dims than `indices2d`.
        """
        if isinstance(indices2d, list):
            indices2d = stack([stack(row, spatial('v')) for row in indices2d], spatial('u'))
        mask_per_part = mask is not None and (non_spatial(indices2d) in mask.shape or self.batch_dims)
        if self.source_idx is not None:
            source_idx = source_idx[self.source_face_shape.name_list]
        for strip in (non_spatial(indices2d) - self.batch_dims).meshgrid():
            indices_np = indices2d[strip].numpy([*spatial(indices2d), self.batch_dims])
            v00 = indices_np[:-1, :-1, :]
            v01 = indices_np[:-1, 1:, :]
            v10 = indices_np[1:, :-1, :]
            v11 = indices_np[1:, 1:, :]
            flip_strip = flip[strip] if isinstance(flip, Tensor) else flip
            if mask is not None and mask_per_part:  # Cannot have different #quads -> add zero-area quad (v00,v00,v00,v00)
                m = mask[strip].numpy([*(spatial(indices2d)-1), self.batch_dims])
                v01 = np.where(m, v01, v00)
                v10 = np.where(m, v10, v00)
                v11 = np.where(m, v11, v00)
            lists = np.stack((v00, v01, v11, v10) if flip_strip else (v00, v10, v11, v01), axis=-1)
            lists = lists.reshape((-1, self.batch_dims.volume, 4))
            if mask is not None and not mask_per_part:
                m = mask.numpy([(spatial(indices2d)-1)])
                lists = lists[np.where(m)[0]]
            self.elements.extend(lists)
            if self.source_idx is not None:
                self.source_idx.extend(source_idx[strip].numpy([spatial(indices2d)-1, self.batch_dims, channel]))
                assert len(self.source_idx) == len(self.elements)

    def add_tris(self, index0: Tensor, indices1d: Tensor, source_idx: Tensor, /, flip: Union[Tensor, bool] = False):
        """

        Args:
            index0: Vertex index shared by all triangles, shape `(...)`.
            indices1d: 1D strip of vertex indices, shape `(..., strip:s)`. Neighbors are connected with `index0` to form triangles.
            source_idx:
            flip:
        """
        if self.source_idx is not None:
            source_idx = source_idx[self.source_face_shape.name_list]
        for tri in (index0.shape - self.batch_dims).meshgrid():
            idx_np = indices1d[tri].numpy([spatial, self.batch_dims])
            v1 = idx_np[:-1, :]
            v2 = idx_np[1:, :]
            v0_ = index0[tri].numpy([self.batch_dims])[None, :].repeat(v1.shape[0], axis=0)
            flip_tri = flip[tri] if isinstance(flip, Tensor) else flip
            lists = np.stack((v0_, v1, v2) if flip_tri else (v0_, v2, v1), axis=-1)
            self.elements.extend(lists.reshape((-1, self.batch_dims.volume, 3)))
            if self.source_idx is not None:
                self.source_idx.extend(source_idx[tri].numpy([spatial(indices1d)-1, self.batch_dims, channel]))
                assert len(self.source_idx) == len(self.elements)

    def triangulate(self, loop_indices: Tensor, /, flip: Union[Tensor, bool] = False):
        """
        Fills a closed loop of vertices by triangulating it.
        The triangles are chosen to have similar angles (large minimum, small maximum), and short edge lengths.

        Args:
            loop_indices: Ordered sequence(s) of vertex ids forming closed loops.
            flip: Whether to flip the triangle orientations, i.e. reverse the order in which the vertices are listed.
        """
        if isinstance(loop_indices, list):
            loop_indices = stack(loop_indices, spatial('loop'))
        for strip in (non_spatial(loop_indices) - self.batch_dims).meshgrid():
            loop_np = loop_indices[strip].numpy([spatial, self.batch_dims])
            if isinstance(flip, Tensor):
                flip_strip = flip[strip]
                flip_np = np.asarray(flip_strip.numpy([self.batch_dims]), dtype=bool).reshape(-1)
            else:
                flip_np = np.full(self.batch_dims.volume, bool(flip), dtype=bool)
            triangles_per_batch = []
            for bi in range(self.batch_dims.volume):
                triangles_per_batch.append(_triangulate_loop(self.v_buffer[bi, loop_np[:, bi], :], loop_np[:, bi], flip=bool(flip_np[bi])))
            triangles_per_batch = np.asarray(triangles_per_batch, dtype=np.int32)
            assert len({tri.shape[0] for tri in triangles_per_batch}) == 1, "Triangulation must yield the same triangle count for all batches"
            self.elements.extend(np.swapaxes(triangles_per_batch, 0, 1))

    def debug_show(self, edges=True, normals=True):
        from ..field import PointCloud
        from ..vis import show
        mesh = self.build_mesh()
        plot = [mesh, mesh.vertices]
        if edges:
            plot.append(mesh.vertex_graph)
        if normals:
            plot.append(PointCloud(mesh.center, mesh.normals * .05))  # type: ignore[arg-type]
        show(plot, overlay='list')
