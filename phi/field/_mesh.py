from typing import Union, Any

from ._field import SampledField
from .. import math
from ..geom import Geometry, Box
from ..geom._stack import GeometryStack
from ..math import Tensor, Extrapolation, expand, wrap, non_batch, Shape, all_available, instance, spatial, dual
from ..math.magic import slicing_dict


class Mesh(SampledField):

    def __init__(self,
                 elements: Union[Tensor, Geometry],
                 connections: Tensor,
                 values: Any = 1.,
                 extrapolation: Union[Extrapolation, float] = 0.,
                 bounds: Box = None):
        SampledField.__init__(self, elements, expand(wrap(values), non_batch(elements).non_channel), extrapolation, bounds)
        assert not spatial(elements), f"Mesh does not support spatial dimensions but got elements with shape {elements.shape}"
        assert isinstance(connections, Tensor), f"connections must be a Tensor but got {type(connections)}"
        assert instance(elements) in instance(connections), f"Element instance dim {instance(elements)} must be present on connections but got {connections.shape}"
        assert dual(connections).rank == instance(connections).rank, f"Connections must contain one dual dimension for every instance dimension but got shape {connections.shape}"
        self._connections = connections

    @property
    def shape(self):
        return self._elements.shape.without('vector') & self._values.shape

    @property
    def connections(self):
        return self._connections

    def __getitem__(self, item):
        if instance(self._elements).only(tuple(item)):
            raise NotImplementedError("Slicing along instance dimensions not yet supported")
        item = slicing_dict(self, item)
        if not item:
            return self
        elements = self.elements[{dim: selection for dim, selection in item.items() if dim != 'vector'}]
        connections = self._connections[item]
        values = self._values[item]
        extrapolation = self._extrapolation[item]
        return Mesh(elements, connections, values, extrapolation, self._bounds)

    def with_values(self, values):
        return Mesh(self.elements, self._connections, values, self._extrapolation, self._bounds)

    def with_extrapolation(self, extrapolation: Extrapolation):
        return Mesh(self.elements, self._connections, self._values, extrapolation, self._bounds)

    def with_bounds(self, bounds: Box):
        return Mesh(self.elements, self._connections, self._values, self._extrapolation, bounds)

    def __value_attrs__(self):
        return '_values', '_extrapolation'

    def __variable_attrs__(self):
        return '_values', '_elements', '_connections'

    def __expand__(self, dims: Shape, **kwargs) -> 'PointCloud':
        return self.with_values(expand(self.values, dims, **kwargs))

    def __eq__(self, other):
        if not type(self) == type(other):
            return False
        # Check everything but __variable_attrs__ (values): elements type, extrapolation, add_overlapping
        if type(self.elements) is not type(other.elements):
            return False
        if self.extrapolation != other.extrapolation:
            return False
        if self._add_overlapping != other._add_overlapping:
            return False
        if self.values is None:
            return other.values is None
        if other.values is None:
            return False
        if not all_available(self.values) or not all_available(other.values):  # tracers involved
            if all_available(self.values) != all_available(other.values):
                return False
            else:  # both tracers
                return self.values.shape == other.values.shape
        if not all_available(self._connections) or not all_available(other._connections):
            if all_available(self._connections) != all_available(other._connections):
                return False
            else:  # both tracers
                return self._connections.shape == other._connections.shape
        return bool((self.values == other.values).all)

    @property
    def bounds(self) -> Box:
        if self._bounds is not None:
            return self._bounds
        else:
            from phi.field._field_math import data_bounds
            bounds = data_bounds(self.elements.center)
            radius = math.max(self.elements.bounding_radius())
            return Box(bounds.lower - radius, bounds.upper + radius)

    def _sample(self, geometry: Geometry, **kwargs) -> math.Tensor:
        if geometry == self.elements:
            return self.values
        if isinstance(geometry, GeometryStack):
            sampled = [self._sample(g, **kwargs) for g in geometry.geometries]
            return math.stack(sampled, geometry.geometries.shape)
        raise NotImplementedError("Interpolation not yet implemented")

    def __repr__(self):
        try:
            return "Mesh[%s]" % (self.shape,)
        except:
            return "Mesh[invalid]"


def connected_distances(mesh: Mesh):
    con = mesh.connections
    from phi.math._sparse import CompressedSparseMatrix, SparseCoordinateTensor
    if isinstance(con, CompressedSparseMatrix):
        con = con.decompress()
    inst_dim = instance(mesh.elements).name
    origin = mesh.points[{inst_dim: con._indices[inst_dim]}]
    target = mesh.points[{inst_dim: con._indices['~'+inst_dim]}]
    dx = target - origin
    return con._with_values(dx)
