import warnings

from phi import math
from ._geom import Geometry, NO_GEOMETRY
from ._box import bounding_box, Box
from ..math._shape import merge_shapes
from ..math._magic_ops import variable_attributes, copy_with
from ..math.magic import PhiTreeNode


class Union(Geometry):

    def __init__(self, geometries):
        self._geometries = tuple(geometries)
        assert len(self._geometries) > 0
        for g in self._geometries[1:]:
            assert g.spatial_rank == self._geometries[0].spatial_rank
        self._shape = merge_shapes(*[g.shape for g in geometries])

    @property
    def shape(self):
        return self._shape

    @property
    def geometries(self):
        return self._geometries

    @property
    def rank(self):
        return self.geometries[0].spatial_rank

    def lies_inside(self, location):
        return math.any([geometry.lies_inside(location) for geometry in self.geometries], dim='0')

    def approximate_signed_distance(self, location):
        return math.min([geometry.approximate_signed_distance(location) for geometry in self.geometries], dim='0')

    @property
    def center(self):
        return self._bounding_box().center

    @property
    def volume(self) -> math.Tensor:
        warnings.warn("Volume of a union assumes geometries do not overlap and may not be accurate otherwise.", RuntimeWarning)
        return math.sum([g.volume for g in self.geometries], dim='0')

    @property
    def shape_type(self) -> math.Tensor:
        return math.tensor('?')

    def bounding_radius(self):
        return self._bounding_box().bounding_radius()

    def bounding_half_extent(self):
        return self._bounding_box().bounding_half_extent()

    def _bounding_box(self):
        boxes = [bounding_box(g) for g in self.geometries]
        lower = math.min([b.lower for b in boxes], dim='0')
        upper = math.max([b.upper for b in boxes], dim='0')
        return Box(lower, upper)

    def shifted(self, delta) -> Geometry:
        return Union([geometry.shifted(delta) for geometry in self.geometries])

    def rotated(self, angle) -> Geometry:
        from ._transform import rotate
        return rotate(self, angle)


def union(*geometries) -> Geometry:
    """
    Union of the given geometries.
    A point lies inside the union if it lies within at least one of the geometries.

    Args:
      geometries: arbitrary geometries with same spatial dims. Arbitrary batch dims are allowed.
      *geometries: 

    Returns:
      union Geometry

    """
    if len(geometries) == 1 and isinstance(geometries[0], (tuple, list)):
        geometries = geometries[0]
    if len(geometries) == 0:
        return NO_GEOMETRY
    elif len(geometries) == 1:
        return geometries[0]
    elif all(type(g) == type(geometries[0]) and isinstance(g, PhiTreeNode) for g in geometries):
        attrs = variable_attributes(geometries[0])
        values = {a: math.stack([getattr(g, a) for g in geometries], math.instance('union')) for a in attrs}
        return copy_with(geometries[0], **values)
    else:
        base_geometries = ()
        for geometry in geometries:
            base_geometries += geometry.geometries if isinstance(geometry, Union) else (geometry,)
        return Union(base_geometries)


Geometry.__add__ = lambda g1, g2: union(g1, g2)
