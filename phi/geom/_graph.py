from typing import Union, Tuple, Dict, Any

from phiml.math import Tensor, Shape, channel, shape, non_batch, dual
from ._geom import Geometry, Point
from .. import math


class Graph(Geometry):
    """
    A graph consists of multiple geometry nodes and corresponding edge information.

    Edges are stored as a Tensor with the same axes ad `geometry` plus their dual counterparts.
    Additional dimensions can be added to `edges` to store vector-valued connectivity weights.
    """

    def __init__(self, nodes: Union[Geometry, Tensor], edges: Tensor, boundary: Dict[str, Dict[str, slice]]):
        if isinstance(nodes, Tensor):
            assert 'vector' in channel(nodes) and channel(nodes).get_item_names('vector') is not None, f"nodes must have a 'vector' dim listing the physical dimensions but got {shape(nodes)}"
        node_dims = non_batch(nodes).non_channel
        assert node_dims in edges.shape and edges.shape.dual.rank == node_dims.rank, f"edges must contain all node dims {node_dims} as primal and dual but got {edges.shape}"
        self._nodes: Geometry = nodes if isinstance(nodes, Geometry) else Point(nodes)
        self._edges = edges
        self._boundary = boundary
        self._deltas = math.pairwise_distances(self._nodes.center, format=edges)
        self._distances = math.vec_length(self._deltas)
        self._connectivity = math.tensor_like(edges, 1) if math.is_sparse(edges) else (edges != 0) & ~math.is_nan(edges)

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
        raise NotImplementedError

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
        return Graph(self.nodes.at(center), self._edges, self._boundary)

    def rotated(self, angle: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError

    def __hash__(self):
        return hash(self._nodes)

    def __getitem__(self, item):
        node_dims = non_batch(self._nodes).non_channel
        edge_sel = {}
        for i, (dim, sel) in enumerate(item.items()):
            if dim in node_dims:
                dual_dim = '~' + dim
                if dual_dim not in self._edges.shape:
                    dual_dim = dual(self._edges).shape.names[i]
                edge_sel[dim] = edge_sel[dual_dim] = sel
        return Graph(self._nodes[item], self._edges[edge_sel], self._boundary)
