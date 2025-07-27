from dataclasses import dataclass
from functools import cached_property
from typing import Union, Tuple, Dict, Any, Optional

from phiml.dataclasses import sliceable, replace
from phiml.math import Tensor, Shape, channel, shape, non_batch, dual, batch
from phiml.math.magic import slicing_dict
from ._functions import vec_length
from ._geom import Geometry, Point
from .. import math


@sliceable(keepdims='vector')
@dataclass(frozen=True, eq=False)
class Graph(Geometry):
    """
    A graph consists of multiple geometry nodes and corresponding edge information.

    Edges are stored as a Tensor with the same axes ad `geometry` plus their dual counterparts.
    Additional dimensions can be added to `edges` to store vector-valued connectivity weights.
    """
    nodes: Geometry
    edges: Tensor
    boundary: Dict[str, Dict[str, slice]]

    variable_attrs = ('nodes', 'edges')

    def __post_init__(self):
        assert isinstance(self.nodes, Geometry), f"nodes must be a Geometry  but got {self.nodes}"
        node_dims = non_batch(self.nodes).non_channel
        assert node_dims in self.edges.shape and self.edges.shape.dual.rank == node_dims.rank, f"edges must contain all node dims {node_dims} as primal and dual but got {self.edges.shape}"

    @cached_property
    def connectivity(self) -> Tensor:
        return math.tensor_like(self.edges, 1) if math.is_sparse(self.edges) else (self.edges != 0) & ~math.is_nan(self.edges)

    def as_points(self):
        return replace(self, nodes=Point(self.nodes.center))

    @cached_property
    def deltas(self):
        return math.pairwise_distances(self.nodes.center, format=self.edges)

    @cached_property
    def unit_deltas(self):
        return math.safe_div(self.deltas, self.distances)

    @cached_property
    def distances(self):
        return vec_length(self.deltas)

    @cached_property
    def bounding_distance(self) -> Optional[Tensor]:
        return math.max(self.distances)

    @property
    def center(self) -> Tensor:
        return self.nodes.center

    @property
    def shape(self) -> Shape:
        return self.nodes.shape

    @property
    def volume(self) -> Tensor:
        return self.nodes.volume

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
        return self.boundary

    @property
    def boundary_faces(self) -> Dict[Any, Dict[str, slice]]:
        raise NotImplementedError  # connections between boundary elements

    @property
    def face_shape(self) -> Shape:
        return non_batch(self.edges).non_channel

    def lies_inside(self, location: Tensor) -> Tensor:
        raise NotImplementedError

    def approximate_closest_surface(self, location: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError

    def approximate_signed_distance(self, location: Tensor) -> Tensor:
        raise NotImplementedError

    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        raise NotImplementedError

    def bounding_radius(self) -> Tensor:
        return self.nodes.bounding_radius()

    def bounding_half_extent(self) -> Tensor:
        return self.nodes.bounding_half_extent()

    def at(self, center: Tensor) -> 'Geometry':
        raise NotImplementedError("Changing the node positions of a Graph is not supported as it would invalidate distances.")
        # warnings.warn("Changing the node positions of a graph triggers re-evaluation of distances.", RuntimeWarning, stacklevel=2)
        # return Graph(self.nodes.at(center), self._edges, self._boundary, bounding_distance=self._bounding_distance is not None)

    def shifted(self, delta: Tensor) -> 'Geometry':
        if non_batch(delta).non_channel.only(self.nodes.shape):  # shift varies between elements
            raise NotImplementedError("Shifting the node positions of a Graph is not supported as it would invalidate distances.")
        return replace(self, nodes=self.nodes.shifted(delta))

    def rotated(self, angle: Union[float, Tensor]) -> 'Geometry':
        return replace(self, nodes=self.nodes.rotated(angle))

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        return replace(self, nodes=self.nodes.scaled(factor))

    def __getitem__(self, item):
        item = slicing_dict(self, item)
        node_dims = non_batch(self.nodes).non_channel
        edge_sel = {}
        for i, (dim, sel) in enumerate(item.items()):
            if dim in node_dims:
                dual_dim = '~' + dim
                if dual_dim not in self.edges.shape:
                    dual_dim = dual(self.edges).shape.names[i]
                edge_sel[dim] = edge_sel[dual_dim] = sel
            elif dim in batch(self):
                edge_sel[dim] = sel
        return Graph(self.nodes[item], self.edges[edge_sel], self.boundary)


def graph(nodes: Union[Geometry, Tensor],
          edges: Tensor,
          boundary: Dict[str, Dict[str, slice]] = None,
          build_distances=True,  # remaining for legacy reasons. Now evaluated on demand.
          build_bounding_distance=False) -> Graph:
    """
    Construct a `Graph`.

    Args:
        nodes: Location `Tensor` or `Geometry` objects representing the nodes.
        edges: Connectivity and edge value `Tensor`.
        boundary: Named boundary sets.

    Returns:
        `Graph`
    """
    if isinstance(nodes, Tensor):
        assert 'vector' in channel(nodes) and channel(nodes).get_item_names('vector') is not None, f"nodes must have a 'vector' dim listing the physical dimensions but got {shape(nodes)}"
        nodes = Point(nodes)
    boundary = {} if boundary is None else boundary
    return Graph(nodes, edges, boundary)
