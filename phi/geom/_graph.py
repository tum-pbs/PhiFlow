from typing import Union, Tuple, Dict, Any, Optional

from phiml.math import Tensor, Shape, channel, shape, non_batch, dual, batch
from phiml.math.magic import slicing_dict
from ._geom import Geometry, Point
from .. import math


class Graph(Geometry):
    """
    A graph consists of multiple geometry nodes and corresponding edge information.

    Edges are stored as a Tensor with the same axes ad `geometry` plus their dual counterparts.
    Additional dimensions can be added to `edges` to store vector-valued connectivity weights.
    """

    def __init__(self,
                 nodes: Union[Geometry, Tensor],
                 edges: Tensor,
                 boundary: Dict[str, Dict[str, slice]],
                 deltas: Optional[Tensor] = None,
                 distances: Optional[Tensor] = None,
                 bounding_distance: Union[Tensor, float, None] = None):
        """
        Create a graph where `nodes` are connected by `edges`.

        Args:
            nodes: `Geometry` collection or `Tensor` to denote points.
            edges: Edge weight matrix. Must have the instance and spatial dims of `nodes` plus their dual counterparts.
            boundary: Marks ranges of nodes as boundary elements.
            deltas: (Optional) Pre-computed position difference matrix.
            distances: (Optional) Pre-computed distance matrix.
            bounding_distance: (Optional) Pre-computed distance bounds. No distance is larger than this value. If `True`, will be computed now, if `False`, will not be computed.
        """
        assert isinstance(nodes, Geometry), f"nodes must be a Geometry  but got {nodes}"
        node_dims = non_batch(nodes).non_channel
        assert node_dims in edges.shape and edges.shape.dual.rank == node_dims.rank, f"edges must contain all node dims {node_dims} as primal and dual but got {edges.shape}"
        self._nodes: Geometry = nodes if isinstance(nodes, Geometry) else Point(nodes)
        self._edges = edges
        self._boundary = boundary
        self._deltas = deltas
        self._distances = distances
        self._connectivity = math.tensor_like(edges, 1) if math.is_sparse(edges) else (edges != 0) & ~math.is_nan(edges)
        if isinstance(bounding_distance, bool):
            self._bounding_distance = math.max(self._distances) if bounding_distance else None
        else:
            self._bounding_distance = bounding_distance

    def __variable_attrs__(self):
        return '_nodes', '_edges', '_deltas', '_distances', '_connectivity'

    def __value_attrs__(self):
        return '_nodes',

    @property
    def edges(self):
        return self._edges

    @property
    def connectivity(self) -> Tensor:
        return self._connectivity

    @property
    def nodes(self) -> Geometry:
        return self._nodes

    def as_points(self):
        return Graph(Point(self._nodes.center), self._edges, self._boundary, self._deltas, self._distances, self._bounding_distance)

    @property
    def deltas(self):
        return self._deltas

    @property
    def unit_deltas(self):
        return math.safe_div(self._deltas, self._distances)

    @property
    def distances(self):
        return self._distances

    @property
    def bounding_distance(self) -> Optional[Tensor]:
        return self._bounding_distance

    @property
    def center(self) -> Tensor:
        return self._nodes.center

    @property
    def shape(self) -> Shape:
        return self._nodes.shape

    @property
    def volume(self) -> Tensor:
        return self._nodes.volume

    @property
    def faces(self) -> 'Geometry':
        raise NotImplementedError

    @property
    def face_centers(self) -> Tensor:
        raise NotImplementedError

    @property
    def face_areas(self) -> Tensor:
        raise NotImplementedError

    @property
    def face_normals(self) -> Tensor:
        raise NotImplementedError

    @property
    def boundary_elements(self) -> Dict[str, Dict[str, slice]]:
        return self._boundary

    @property
    def boundary_faces(self) -> Dict[Any, Dict[str, slice]]:
        raise NotImplementedError  # connections between boundary elements

    @property
    def face_shape(self) -> Shape:
        return non_batch(self._edges).non_channel

    def lies_inside(self, location: Tensor) -> Tensor:
        raise NotImplementedError

    def approximate_closest_surface(self, location: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError

    def approximate_signed_distance(self, location: Tensor) -> Tensor:
        raise NotImplementedError

    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        raise NotImplementedError

    def bounding_radius(self) -> Tensor:
        return self._nodes.bounding_radius()

    def bounding_half_extent(self) -> Tensor:
        return self._nodes.bounding_half_extent()

    def at(self, center: Tensor) -> 'Geometry':
        raise NotImplementedError("Changing the node positions of a Graph is not supported as it would invalidate distances.")
        # warnings.warn("Changing the node positions of a graph triggers re-evaluation of distances.", RuntimeWarning, stacklevel=2)
        # return Graph(self.nodes.at(center), self._edges, self._boundary, bounding_distance=self._bounding_distance is not None)

    def shifted(self, delta: Tensor) -> 'Geometry':
        if non_batch(delta).non_channel.only(self._nodes.shape) and self._deltas is not None:  # shift varies between elements
            raise NotImplementedError("Shifting the node positions of a Graph is not supported as it would invalidate distances.")
        return Graph(self.nodes.shifted(delta), self._edges, self._boundary, deltas=self._deltas, distances=self._distances, bounding_distance=self._bounding_distance is not None)

    def rotated(self, angle: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError

    def __getitem__(self, item):
        item = slicing_dict(self, item)
        node_dims = non_batch(self._nodes).non_channel
        edge_sel = {}
        for i, (dim, sel) in enumerate(item.items()):
            if dim in node_dims:
                dual_dim = '~' + dim
                if dual_dim not in self._edges.shape:
                    dual_dim = dual(self._edges).shape.names[i]
                edge_sel[dim] = edge_sel[dual_dim] = sel
            elif dim in batch(self):
                edge_sel[dim] = sel
        deltas = self._deltas[edge_sel] if self._deltas is not None else None
        distances = self._distances[edge_sel] if self._distances is not None else None
        bounding_distance = self._bounding_distance[item] if self._bounding_distance is not None else None
        return Graph(self._nodes[item], self._edges[edge_sel], self._boundary, deltas, distances, bounding_distance)


def graph(nodes: Union[Geometry, Tensor],
          edges: Tensor,
          boundary: Dict[str, Dict[str, slice]] = None,
          build_distances=True,
          build_bounding_distance=False) -> Graph:
    """
    Construct a `Graph`.

    Args:
        nodes: Location `Tensor` or `Geometry` objects representing the nodes.
        edges: Connectivity and edge value `Tensor`.
        boundary: Named boundary sets.
        build_distances: Whether to compute all edge lengths.
            This enables the properties `Graph.deltas`, `Graph.unit_deltas`, `Graph.distances`.
        build_bounding_distance: Whether to compute the maximum edge length.
            This enables the property `Graph.bounding_distance`.

    Returns:
        `Graph`
    """
    if isinstance(nodes, Tensor):
        assert 'vector' in channel(nodes) and channel(nodes).get_item_names('vector') is not None, f"nodes must have a 'vector' dim listing the physical dimensions but got {shape(nodes)}"
        nodes = Point(nodes)
    boundary = {} if boundary is None else boundary
    deltas = math.pairwise_distances(nodes.center, format=edges) if build_distances else None
    distances = math.vec_length(deltas) if build_distances else None
    bound = math.max(distances) if build_bounding_distance else None
    return Graph(nodes, edges, boundary, deltas, distances, bound)
