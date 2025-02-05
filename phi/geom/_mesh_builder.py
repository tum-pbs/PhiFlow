from typing import Union, Dict

import numpy as np

from phiml.math import Tensor, range_tensor, non_spatial, spatial, instance, Shape, EMPTY_SHAPE, math, stack, channel, expand, wrap
from ._mesh import Mesh, mesh_from_numpy


class MeshBuilder:
    def __init__(self, element_rank: int, batch_dims: Shape = None, source_face_shape: Shape = None):
        self.element_rank = element_rank
        self.batch_dims = batch_dims or EMPTY_SHAPE
        self.axes = None
        self.v_buffer = np.empty((batch_dims.volume, 0, 3))
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

    def add_quads(self, indices2d: Tensor, source_idx: Tensor, /, flip: Union[Tensor, bool] = False):
        """
        Add quads to the mesh, connecting previously added vertices.

        Args:
            indices2d: 2D tensor of vertex indices, shape `(..., u:s, v:s)`.
                Use `MeshBuilder.vertex_indices()` or the output of `add_vertices()` to get indices of existing vertices.
            source_idx: Meta-information about the added quads, can have fewer dims than `indices2d`.
            flip: Whether to flip the quad orientation, i.e. reverse the order in which the vertices are listed per quad.
                Can have fewer dims than `indices2d`.
        """
        if self.source_idx is not None:
            source_idx = source_idx[self.source_face_shape.name_list]
        for strip in (non_spatial(indices2d) - self.batch_dims).meshgrid():
            indices_np = indices2d[strip].numpy([*spatial(indices2d), self.batch_dims])
            v00 = indices_np[:-1, :-1, :]
            v01 = indices_np[:-1, 1:, :]
            v10 = indices_np[1:, :-1, :]
            v11 = indices_np[1:, 1:, :]
            flip_strip = flip[strip] if isinstance(flip, Tensor) else flip
            lists = np.stack((v00, v01, v11, v10) if flip_strip else (v00, v10, v11, v01), axis=-1)
            self.elements.extend(lists.reshape((-1, self.batch_dims.volume, 4)))
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

    def debug_show(self, normals=True):
        from phi.field import PointCloud
        from phi.vis import show
        mesh = self.build_mesh()
        plot = [mesh, mesh.vertices]
        if normals:
            plot.append(PointCloud(mesh.center, mesh.normals * .05))
        show(plot, overlay='list')
