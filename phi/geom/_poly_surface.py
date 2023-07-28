from typing import Tuple, Dict

import numpy as np

from .. import math
from phiml.math import Tensor, Shape, channel, NUMPY, shape, instance, dual, rename_dims, expand, spatial, pack_dims, wrap, sparse_tensor, vec
from ._geom import Geometry


class PolygonSurface(Geometry):

    def __init__(self, vertices: Tensor, polygons: Tensor, vertex_count: float or Tensor):
        """
        Args:
            vertices: Vertex positions, shape (vertices:i, vector:c)
                Vertex 0 must be at position 0.
            polygons: `Tensor` containing vertex indices, Shape (polygons:i, vertex_index:s).
                This can be a sparse or dense tensor.
                Invalid indices (index >= vertex_count) must still represent existing vertices. (or -1?)
            vertex_count: Number of vertices per polygon, shape (polygons,)
        """
        assert instance(vertices).rank == 1, f"vertices must have exactly one instance dimension but got {shape(vertices)}"
        vertices = rename_dims(vertices, instance, 'vertices')
        assert 'vector' in channel(vertices) and channel(vertices).get_item_names('vector') is not None, "vertices must have a 'vector' dim listing the physical dimensions"
        vertex_count = expand(vertex_count, instance(polygons))
        assert 'vertex_index' in spatial(polygons), f"polygons must have exactly one spatial dimension called 'vertex_index' but got {shape(polygons)}"
        assert polygons.dtype.kind == int, f"polygons must be integer lists but got dtype {polygons.dtype}"
        self._vertices = vertices
        self._polygons = polygons
        self._vertex_count = vertex_count

    @property
    def shape(self) -> Shape:
        return shape(self._polygons).non_spatial & channel(self._vertices)

    def __getitem__(self, item):
        assert 'vertices' not in item, "Cannot slice PolygonSurface along 'vertices'"
        assert 'vertex_index' not in item, "Cannot slice PolygonSurface along 'vertex_index'"
        vertices = self._vertices[item]
        polygons = self._polygons[item]
        vertex_count = self._vertex_count[item]
        return PolygonSurface(vertices, polygons, vertex_count)

    @property
    def center(self) -> Tensor:
        # stack 0 to vertices
        # polygons+1
        vertex_pos = self._vertices[self._polygons]
        vertex_pos *= self._valid_mask()
        return math.sum(vertex_pos, 'vertex_index') / self._vertex_count

    def corners(self):
        return self._vertices[self._polygons]

    def _valid_mask(self):
        max_index = spatial(self._polygons).size
        with NUMPY:
            valid = math.range(spatial(vertex_index=max_index)) < self._vertex_count
        return valid

    def edges(self, bidirectional=False) -> Tuple[Tensor, Tensor]:
        last_vertex = expand(self._polygons.vertex_index[self._vertex_count-1], spatial(vertex_index=1))
        vertex1 = math.concat([last_vertex, self._polygons.vertex_index[:-1]], 'vertex_index')
        vertex2 = self._polygons
        indices = math.vec(from_vertex=vertex1, to_vertex=vertex2)
        indices = pack_dims(indices, ['vertex_index', instance(self._polygons)], instance('edges'))
        vdim = list(instance(self._vertices).names)
        indices = rename_dims(indices, 'vector', channel(vector=['~'+s for s in vdim] + vdim))
        valid = self._valid_mask()
        values = pack_dims(valid, 'vertex_index', instance('edges'))  # ToDo pack
        dense_shape = dual(vertices=self._vertices.vertices.size) & instance(self._vertices)
        edges = sparse_tensor(indices, values, dense_shape, can_contain_double_entries=True, indices_sorted=False)
        # ToDo remove zero values
        if bidirectional:
            pass  # ToDo add transpose to make matrix symmetric, then remove doubles
        # ToDo remove doubles
        return self._vertices, edges

    def connected_polygons(self, min_shared_vertices: int or None = 2, self_connections = False) -> Tensor:
        """
        Counts the shared vertices between all pairs of polygons.

        Args:
            min_shared_vertices: If not `None`, drops polygon pairs that share fewer than this number of vertices.

        Returns:
            Possibly sparse `Tensor` of shape (~polygons, polygons) storing the number of shared vertices.
        """
        valid = self._valid_mask()
        counts = {}  # (poly_idx, poly_idx) -> count
        for v_idx in range(self._vertices.vertices.size):  # for each vertex, find polygons that neighbor it, add +1 to all pairs
            polygons = (self._polygons == v_idx) * valid
            polygons = math.sum(polygons, 'vertex_index')  # doesn't matter which index that vertex is
            polygon_indices = math.nonzero(polygons)
            for p1 in polygon_indices:
                for p2 in polygon_indices:
                    if p1 != p2 or self_connections:
                        if (p1, p2) in counts:
                            counts[(p1, p2)] += 1
                        else:
                            counts[(p1, p2)] = 1
        if min_shared_vertices is not None:
            counts = {k: v for k, v in counts.items() if v >= min_shared_vertices}
        polygons_name = instance(self._polygons).name
        indices = wrap(tuple(counts), instance('poly_pairs'), channel(vector=['~'+polygons_name, polygons_name]))
        values = wrap(tuple(counts.values()), instance('poly_pairs'))
        dense_shape = dual(**instance(self._polygons).untyped_dict) & instance(self._polygons)
        shared_vertex_count = sparse_tensor(indices, values, dense_shape, can_contain_double_entries=True, indices_sorted=False)
        # if min_shared_vertices is not None:
        #     shared_vertex_count = math.where(shared_vertex_count >= min_shared_vertices, shared_vertex_count, 0)
        return shared_vertex_count

    def faces(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Assembles information about the boundaries of the polygons that make up the surface.
        For 2D polygons, the faces are edges, for 3D polygons, the faces are planar polygons.

        Returns:
            center: Center of face connecting a pair of polygons. Shape (~polygons, polygons, vector).
                Returns 0-vectors for unconnected polygons.
            area: Area of face connecting a pair of polygons. Shape (~polygons, polygons).
                Returns 0 for unconnected polygons.
            normal: Normal vector of face connecting a pair of polygons. Shape (~polygons, polygons, vector).
                Unconnected polygons are assigned the vector 0.
                The vector points out of polygon and into ~polygon.
        """
        raise NotImplementedError

    @property
    def vertices(self):
        return self._vertices

    @property
    def volume(self) -> Tensor:
        raise NotImplementedError

    @property
    def shape_type(self) -> Tensor:
        raise NotImplementedError

    def lies_inside(self, location: Tensor) -> Tensor:
        raise NotImplementedError

    def approximate_signed_distance(self, location: Tensor or tuple) -> Tensor:
        raise NotImplementedError

    def push(self, positions: Tensor, outward: bool = True, shift_amount: float = 0) -> Tensor:
        raise NotImplementedError

    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        raise NotImplementedError

    def bounding_radius(self) -> Tensor:
        center = self.center
        vertex_pos = self._vertices[self._polygons]
        max_dist = math.max(math.vec_length(vertex_pos - center), 'vertex_index')
        return max_dist

    def bounding_half_extent(self) -> Tensor:
        raise NotImplementedError

    def at(self, center: Tensor) -> 'Geometry':
        raise NotImplementedError

    def rotated(self, angle: float or Tensor) -> 'Geometry':
        raise NotImplementedError

    def scaled(self, factor: float or Tensor) -> 'Geometry':
        raise NotImplementedError

    def __hash__(self):
        return hash(self._vertices) + hash(self._polygons)


def surface_from_numpy(points, elements) -> PolygonSurface:
    points = np.asarray(points)
    dim = points.shape[-1]
    if dim == 2:
        vertices = vec(x=points[:, 0].tolist(), y=points[:, 1].tolist())
    elif dim == 3:
        vertices = vec(x=points[:, 0].tolist(), y=points[:, 1].tolist(), z=points[:, 2].tolist())
    else:
        raise NotImplementedError(f"dim={dim} not supported")
    try:
        elements_np = np.stack(elements).astype(np.int32)
        vertex_count = elements_np.shape[-1]
    except ValueError:
        vertex_count = wrap([len(e) for e in elements], instance('polygons'))
        max_len = vertex_count.max
        elements_np = np.zeros((len(elements), max_len), dtype=np.int32)
        for i, element in enumerate(elements):
            elements_np[i, :len(element)] = element
    polygons = wrap(elements_np, instance('polygons'), spatial('vertex_index'))
    return PolygonSurface(vertices, polygons, vertex_count=vertex_count)


def load_su2(file_or_mesh: str) -> Tuple[PolygonSurface, Dict[str, Tensor]]:
    """
    Loads an unstructured mesh from a `.su2` file.

    Args:
        file_or_mesh: Path to `.su2` file or *ezmesh* `Mesh` instance.

    Returns:
        surface: `PolygonSurface`
        markers: Edges/Faces marked
            sparse (vertices, vertices) -> int
    """
    if isinstance(file_or_mesh, str):
        from ezmesh import import_from_file
        mesh = import_from_file(file_or_mesh)
    else:
        mesh = file_or_mesh
    surface = surface_from_numpy(mesh.points, mesh.elements)
    vertices = surface.vertices
    markers = {}
    for name, pair_list in mesh.markers.items():
        marker_indices = wrap(np.stack(pair_list).astype(np.int32), instance('edges'), channel(vector='~vertices,vertices'))
        marker_values = expand(True, instance(marker_indices))
        marker_shape = dual(vertices=instance(vertices).size) & instance(vertices=instance(vertices).size)
        marker = sparse_tensor(marker_indices, marker_values, marker_shape, can_contain_double_entries=False, indices_sorted=False)
        markers[name] = marker
    return surface, markers
