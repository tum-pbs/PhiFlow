from typing import Tuple

from .. import math
from ..math import Tensor, Shape, channel, NUMPY, shape, instance, dual, rename_dims, expand, spatial, pack_dims, wrap, sparse_tensor
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


def load_su2() -> Tuple[PolygonSurface, Tensor]:
    """

    Returns:
        surface: `PolygonSurface`
        markers: Edges/Faces marked
            sparse (vertices, vertices) -> int
    """
    import ezmesh
    return surface, markers
