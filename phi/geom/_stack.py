from phi import math
from ._geom import Geometry
from ..math._shape import shape_stack, Shape, BATCH_DIM, _infer_dim_type_from_name


class GeometryStack(Geometry):

    def __init__(self, geometries: tuple or list, dim_name: str, dim_type: str):
        self.geometries = tuple(geometries)
        self.stack_dim_name = dim_name
        self.stack_dim_type = dim_type
        self._shape = shape_stack(dim_name, dim_type, *[g.shape for g in geometries])

    def unstack(self, dimension):
        if dimension == self.stack_dim_name:
            return self.geometries
        else:
            return GeometryStack([g.unstack(dimension) for g in self.geometries], self.stack_dim_name, self.stack_dim_type)

    @property
    def center(self):
        centers = [g.center for g in self.geometries]
        return math.batch_stack(centers, self.stack_dim_name)

    @property
    def shape(self):
        return self._shape

    def lies_inside(self, location: math.Tensor):
        if self.stack_dim_name in location.shape:
            location = location.unstack(self.stack_dim_name)
        else:
            location = [location] * len(self.geometries)
        inside = [g.lies_inside(loc) for g, loc in zip(self.geometries, location)]
        return math.batch_stack(inside, self.stack_dim_name)

    def approximate_signed_distance(self, location: math.Tensor):
        raise NotImplementedError()

    def bounding_radius(self):
        radii = [g.bounding_radius() for g in self.geometries]
        return math.batch_stack(radii, self.stack_dim_name)

    def bounding_half_extent(self):
        values = [g.bounding_half_extent() for g in self.geometries]
        return math.batch_stack(values, self.stack_dim_name)

    def shifted(self, delta: math.Tensor):
        deltas = delta.dimension(self.stack_dim_name).unstack(len(self.geometries))
        geometries = [g.shifted(d) for g, d in zip(self.geometries, deltas)]
        return GeometryStack(geometries, self.stack_dim_name, self.stack_dim_type)

    def rotated(self, angle):
        geometries = [g.rotated(angle) for g in self.geometries]
        return GeometryStack(geometries, self.stack_dim_name, self.stack_dim_type)

    def __eq__(self, other):
        return isinstance(other, GeometryStack) \
               and self._shape == other.shape \
               and self.stack_dim_name == other.stack_dim_name \
               and self.geometries == other.geometries

    def shallow_equals(self, other):
        if self is other:
            return True
        if not isinstance(other, GeometryStack) or self._shape != other.shape:
            return False
        if self.stack_dim_name != other.stack_dim_name:
            return False
        return all(g1.shallow_equals(g2) for g1, g2 in zip(self.geometries, other.geometries))

    def __hash__(self):
        return hash(self.geometries)


def stack(*geometries: Geometry, dim: str):
    return GeometryStack(geometries, dim, _infer_dim_type_from_name(dim))
