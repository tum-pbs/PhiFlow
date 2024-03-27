from typing import Union, Tuple, Dict, Any

from phiml.math import Tensor, Shape, channel, shape
from ._geom import Geometry, Point
from .. import math


class Graph(Geometry):
    """
    A graph consists of multiple geometries and their connectivity information.
    """

    def __init__(self, nodes: Union[Geometry, Tensor], connectivity: Tensor, boundary: Dict[str, Dict[str, slice]]):
        if isinstance(nodes, Tensor):
            assert 'vector' in channel(nodes) and channel(nodes).get_item_names('vector') is not None, f"nodes must have a 'vector' dim listing the physical dimensions but got {shape(nodes)}"
        self._nodes: Geometry = nodes if isinstance(nodes, Geometry) else Point(nodes)
        self._connectivity = connectivity
        self._boundary = boundary
        self._deltas = math.pairwise_distances(self._nodes.center, format=connectivity)
        self._distances = math.vec_length(self._deltas)

    def __variable_attrs__(self):
        return '_nodes', '_deltas', '_distances'

    def __value_attrs__(self):
        return '_nodes',

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
        return Graph(self.nodes.at(center), self._connectivity, self._boundary)

    def rotated(self, angle: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError

    def __hash__(self):
        return hash(self._nodes)

    def __getitem__(self, item):
        return Graph(self._nodes[item], self._connectivity[item], self._boundary)
