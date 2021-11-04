from typing import List

from phi import math
from ._geom import Geometry
from ..math import Tensor
from ..math._shape import shape_stack, Shape, INSTANCE_DIM
from ..math._tensors import variable_attributes, copy_with


class GeometryStack(Geometry):

    def __init__(self, geometries: tuple or list, stack_dim: Shape):
        self.geometries = tuple(geometries)
        self.stack_dim = stack_dim.with_size(len(geometries))
        self._shape = shape_stack(self.stack_dim, *[g.shape for g in geometries])

    def unstack(self, dimension):
        if dimension == self.stack_dim.name:
            return self.geometries
        else:
            return GeometryStack([g.unstack(dimension) for g in self.geometries], self.stack_dim)

    @property
    def center(self):
        centers = [g.center for g in self.geometries]
        return math.stack(centers, self.stack_dim)

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def volume(self) -> math.Tensor:
        if self.stack_dim.type == INSTANCE_DIM:
            raise NotImplementedError("instance dimensions not yet supported")
        return math.stack([g.volume for g in self.geometries], self.stack_dim)

    def lies_inside(self, location: math.Tensor):
        if self.stack_dim in location.shape:
            location = location.unstack(self.stack_dim.name)
        else:
            location = [location] * len(self.geometries)
        inside = [g.lies_inside(loc) for g, loc in zip(self.geometries, location)]
        return math.stack(inside, self.stack_dim)

    def approximate_signed_distance(self, location: math.Tensor):
        raise NotImplementedError()

    def bounding_radius(self):
        radii = [g.bounding_radius() for g in self.geometries]
        return math.stack(radii, self.stack_dim)

    def bounding_half_extent(self):
        values = [g.bounding_half_extent() for g in self.geometries]
        return math.stack(values, self.stack_dim)

    def shifted(self, delta: math.Tensor):
        deltas = delta.dimension(self.stack_dim).unstack(len(self.geometries))
        geometries = [g.shifted(d) for g, d in zip(self.geometries, deltas)]
        return GeometryStack(geometries, self.stack_dim)

    def rotated(self, angle):
        geometries = [g.rotated(angle) for g in self.geometries]
        return GeometryStack(geometries, self.stack_dim)

    def push(self, positions: Tensor, outward: bool = True, shift_amount: float = 0) -> Tensor:
        raise NotImplementedError('GeometryStack.push() is not yet implemented.')

    def __eq__(self, other):
        return isinstance(other, GeometryStack) \
               and self._shape == other.shape \
               and self.stack_dim == other.stack_dim \
               and self.geometries == other.geometries

    def shallow_equals(self, other):
        if self is other:
            return True
        if not isinstance(other, GeometryStack) or self._shape != other.shape:
            return False
        if self.stack_dim != other.stack_dim:
            return False
        return all(g1.shallow_equals(g2) for g1, g2 in zip(self.geometries, other.geometries))

    def __hash__(self):
        return hash(self.geometries)


def stack(geometries: List[Geometry], dim: Shape):
    """ Stacks `geometries` along `dim`. The size of `dim` is ignored. """
    if all(type(g) == type(geometries[0]) for g in geometries):
        attrs = variable_attributes(geometries[0])
        new_attributes = {a: math.stack([getattr(g, a) for g in geometries], dim) for a in attrs}
        return copy_with(geometries[0], **new_attributes)
    return GeometryStack(geometries, dim)
