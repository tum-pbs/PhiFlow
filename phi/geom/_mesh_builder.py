import numpy as np

from phiml.math import Tensor, range_tensor, non_spatial, spatial, instance
from ._mesh import Mesh, mesh_from_numpy


class MeshBuilder:
    def __init__(self, element_rank: int = None):
        self.element_rank = element_rank
        self.axes = None
        self.v_buffer = np.empty((0, 3))
        self.v_positions = dict[str, Tensor]()
        self.v_indices = dict[str, Tensor]()
        self.elements = []

    def build_mesh(self, element_dim=instance('elements')) -> Mesh:
        return mesh_from_numpy(self.v_buffer, self.elements, {}, self.element_rank, axes=self.axes, cell_dim=element_dim)

    def add_vertices(self, name: str, points: Tensor):
        if self.axes is None:
            self.axes = points.vector.item_names
        s = points.shape - 'vector'
        idx = self.v_buffer.shape[0] + range_tensor(s)
        self.v_indices[name] = idx
        self.v_positions[name] = points
        self.v_buffer = np.concatenate([self.v_buffer, points.numpy([s, 'vector'])])
        return idx

    def vertices(self, name: str) -> Tensor:
        return self.v_positions[name]

    def vertex_indices(self, name: str) -> Tensor:
        return self.v_indices[name]

    def new_quads(self, name: str, points: Tensor, /, flip: Tensor | bool = False):
        indices = self.add_vertices(name, points)
        self.add_quads(indices, flip=flip)
        return indices

    def add_quads(self, indices2d: Tensor, /, flip: Tensor | bool = False):
        if self.element_rank is None:
            self.element_rank = 2
        result = []
        for strip in non_spatial(indices2d).meshgrid():
            indices_np = indices2d[strip].numpy(spatial(indices2d))
            v00 = indices_np[:-1, :-1]
            v01 = indices_np[:-1, 1:]
            v10 = indices_np[1:, :-1]
            v11 = indices_np[1:, 1:]
            flip_strip = flip[strip] if isinstance(flip, Tensor) else flip
            lists = np.stack((v00, v01, v11, v10) if flip_strip else (v00, v10, v11, v01), axis=-1)
            result.extend(lists.reshape((-1, 4)))
        self.elements.extend(result)
        return result

    def add_tris(self, index0: Tensor, indices1d: Tensor, /, flip: Tensor | bool = False):
        if self.element_rank is None:
            self.element_rank = 2
        result = []
        for tri in index0.shape.meshgrid():
            idx_np = indices1d[tri].numpy()
            v1 = idx_np[:-1]
            v2 = idx_np[1:]
            v0_ = index0[tri].numpy()[None].repeat(v1.size)
            flip_tri = flip[tri] if isinstance(flip, Tensor) else flip
            lists = np.stack((v0_, v1, v2) if flip_tri else (v0_, v2, v1), axis=-1)
            result.extend(lists.reshape((-1, 3)))
        self.elements.extend(result)
        return result

    def debug_show(self, normals=True):
        from phi.field import PointCloud
        from phi.vis import show
        mesh = self.build_mesh()
        plot = [mesh, mesh.vertices]
        if normals:
            plot.append(PointCloud(mesh.center, mesh.normals * .05))
        show(plot, overlay='list')
