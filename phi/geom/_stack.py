from typing import List, Tuple

from phi import math
from . import GridCell
from ._geom import Geometry
from phiml.math import Tensor, expand
from phiml.math._shape import shape_stack, Shape, INSTANCE_DIM, non_channel
from phiml.math._magic_ops import variable_attributes, copy_with, unstack
from phiml.math.magic import slicing_dict


class GeometryStack(Geometry):

    def __init__(self, geometries: Tensor):
        self.geometries = geometries
        inner_dims = math.merge_shapes(*[g.shape for g in geometries], allow_varying_sizes=True)
        self._stack_dim = geometries.shape.without(inner_dims)
        self._shape = geometries.shape

    def unstack(self, dimension) -> tuple:
        if dimension == self._stack_dim.name:
            return tuple(self.geometries)
        else:
            # return GeometryStack([g.unstack(dimension) for g in self.geometries], self.geometries.shape)
            raise NotImplementedError()

    @property
    def center(self):
        centers = [g.center for g in self.geometries]
        return math.stack(centers, self._stack_dim)

    @property
    def spatial_rank(self) -> int:
        return next(iter(self.geometries)).spatial_rank

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def volume(self) -> math.Tensor:
        if self._stack_dim.type == INSTANCE_DIM:
            raise NotImplementedError("instance dimensions not yet supported")
        return math.stack([g.volume for g in self.geometries], self._stack_dim)

    @property
    def shape_type(self) -> Tensor:
        types = [g.shape_type for g in self.geometries]
        return math.stack(types, self._stack_dim)

    def lies_inside(self, location: math.Tensor):
        if self._stack_dim in location.shape:
            location = location.unstack(self._stack_dim.name)
        else:
            location = [location] * len(self.geometries)
        inside = [g.lies_inside(loc) for g, loc in zip(self.geometries, location)]
        return math.stack(inside, self._stack_dim)

    def approximate_signed_distance(self, location: math.Tensor):
        raise NotImplementedError()

    def bounding_radius(self):
        radii = [expand(g.bounding_radius(), non_channel(g)) for g in self.geometries]
        return math.stack(radii, self._stack_dim)

    def bounding_half_extent(self):
        values = [expand(g.bounding_half_extent(), non_channel(g)) for g in self.geometries]
        return math.stack(values, self._stack_dim)

    def at(self, center: Tensor) -> 'Geometry':
        geometries = [self.geometries[idx].at(center[idx]) for idx in self._stack_dim.meshgrid()]
        return GeometryStack(math.layout(geometries, self._stack_dim))

    def rotated(self, angle):
        geometries = [g.rotated(angle) for g in self.geometries]
        return GeometryStack(math.layout(geometries, self._stack_dim))

    def push(self, positions: Tensor, outward: bool = True, shift_amount: float = 0) -> Tensor:
        raise NotImplementedError('GeometryStack.push() is not yet implemented.')

    def __eq__(self, other):
        return isinstance(other, GeometryStack) \
               and self._shape == other.shape \
               and self._stack_dim == other.stack_dim \
               and self.geometries == other.geometries

    def shallow_equals(self, other):
        if self is other:
            return True
        if not isinstance(other, GeometryStack) or self._shape != other.shape:
            return False
        if self._stack_dim != other.geometries.shape:
            return False
        return all(g1.shallow_equals(g2) for g1, g2 in zip(self.geometries, other.geometries))

    def __hash__(self):
        return hash(self.geometries)
    
    def __getitem__(self, item):
        selected = self.geometries[slicing_dict(self, item)]
        if isinstance(selected, Geometry):
            return selected
        return GeometryStack(selected)
