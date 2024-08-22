from numbers import Number
import warnings
from typing import Union, Dict, Any, Optional, Tuple

from phi import math
from phiml import math
from phiml.math import wrap
from phiml.math._magic_ops import variable_attributes, copy_with
from phiml.math._shape import shape_stack, Shape
from phiml.math._tensors import object_dims
from phiml.math.magic import PhiTreeNode

from ._box import bounding_box, Box
from ._geom import Geometry, NO_GEOMETRY, rotate
from ._geom import InvertedGeometry
from ..math import Tensor, instance
from ..math.magic import slicing_dict


class GeometryStack(Geometry):
    """
    Represents a tensor of Geometries.
    Instance dimensions represent geometry unions and are reduced.
    """

    def __init__(self, geometries: Union[Tensor, Geometry], set_op: Optional[str]=None):
        """
        Args:
            geometries: Tensor[Geometry] with one or multiple dimensions of any type.
        """
        self._geometries = geometries
        ranks = wrap(math.map(lambda g: g.spatial_rank, geometries, dims=object))
        assert ranks.min == ranks.max, f"Can only stack geometries of the same spatial rank but got ranks {ranks}"
        self._shape = geometries.shape
        if set_op is None:
            set_op = 'union'
        assert set_op in ['union', 'intersection'], f"Set operation must be 'union' or 'intersection' but got {set_op}"
        self._set_op = set_op

    @property
    def geometries(self):
        return self._geometries
    
    @property
    def set_op(self):
        return self._set_op

    def __variable_attrs__(self):
        return '_geometries',

    def __value_attrs__(self):
        return '_geometries',

    @property
    def object_dims(self):
        return object_dims(self._geometries)

    def unstack(self, dimension) -> tuple:
        if dimension == self.geometries.shape.name:
            return tuple(self.geometries)
        else:
            # return GeometryStack([g.unstack(dimension) for g in self.geometries], self.geometries.shape)
            raise NotImplementedError()

    def _bounding_box(self):
        if self._set_op == 'intersection':
            raise NotImplementedError
        boxes = math.map(bounding_box, self._geometries, dims=object)
        lower = math.min(boxes.lower, instance)
        upper = math.max(boxes.upper, instance)
        return Box(lower, upper)

    @property
    def center(self) -> Tensor:
        if self._set_op == 'intersection':
            centers = math.map(lambda g: g.center, self._geometries, dims=object)
            if not instance(centers):
                return centers
            # --- bounding box of smallest volume ---
            warnings.warn("Center of an intersection assumes geometries are subsets of each larger geometry and may not be accurate otherwise.", RuntimeWarning)
            vol = math.map(lambda g: g.volume, self._geometries, dims=object)
            boxes = math.map(bounding_box, self._geometries, dims=object)
            smallest_vol_idx = math.argmin(vol, instance)
            return boxes[smallest_vol_idx].center
        return self._bounding_box().center

    @property
    def weighted_center(self):
        centers = math.map(lambda g: g.center, self._geometries, dims=object)
        if not instance(centers):
            return centers
        # --- volume-weighted mean over instance dimensions ---
        vol = math.map(lambda g: g.volume, self._geometries, dims=object)
        return math.sum(centers * vol, instance) / math.sum(vol, instance)  # ToDo could also return bounding box center

    @property
    def spatial_rank(self) -> int:
        return next(iter(self.geometries)).spatial_rank

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def volume(self) -> math.Tensor:
        if instance(self._geometries):
            if self._set_op == 'intersection':
                warn_msg = "Volume of an intersection assumes geometries are subsets of each larger geometry and may not be accurate otherwise."
            else:
                warn_msg = "Volume of a union assumes geometries do not overlap and may not be accurate otherwise."
            warnings.warn(warn_msg, RuntimeWarning)
        vol = math.map(lambda g: g.volume, self._geometries, dims=object)
        if self._set_op == 'intersection':
            return math.min(vol, instance(self._geometries))
        return math.sum(vol, instance(self._geometries))

    def lies_inside(self, location: math.Tensor):
        inside = math.map(lambda g, l: g.lies_inside(l), self._geometries, location, dims=object)
        if self._set_op == 'intersection':
            return math.all(inside, instance(self._geometries))
        return math.any(inside, instance(self._geometries))

    def approximate_signed_distance(self, location: math.Tensor):
        if self._set_op == 'intersection':
            raise NotImplementedError
        dist = math.map(lambda g, l: g.approximate_signed_distance(l), self._geometries, location, dims=object)
        return math.min(dist, instance(self._geometries))

    def approximate_fraction_inside(self, other_geometry: Geometry, balance: Tensor | Number = 0.5) -> Tensor:
        if self._set_op == 'intersection':
            assert isinstance(other_geometry, Geometry)
            radius = other_geometry.bounding_radius()
            location = other_geometry.center
            distances = math.map(lambda g: g.approximate_signed_distance(location), self._geometries, dims=object)
            inside_fraction = balance - distances / radius
            inside_fraction = math.clip(inside_fraction, 0, 1)
            inside_fraction = math.min(inside_fraction, instance(self._geometries))
            return inside_fraction
        return super().approximate_fraction_inside(other_geometry, balance)

    def approximate_closest_surface(self, location: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self._set_op == 'intersection':
            raise NotImplementedError
        signed_dist, delta, normals, offsets, face_idx = math.map(lambda g, l: g.approximate_closest_surface(l), self._geometries, location, dims=object)
        return math.at_min((signed_dist, delta, normals, offsets, face_idx), key=abs(signed_dist), dim=instance)

    def bounding_radius(self):
        if self._set_op == 'intersection':
            raise NotImplementedError
        center = self.center
        rad = math.map(lambda g: math.vec_length(g.center - center) + g.bounding_radius(), self._geometries, dims=object)
        return math.max(rad, instance(self._geometries))

    def bounding_half_extent(self):
        return self._bounding_box().half_size

    def shifted(self, delta: Tensor) -> 'Geometry':
        return GeometryStack(math.map(lambda g: g.shifted(delta), self._geometries, dims=object), set_op=self._set_op)

    def at(self, center: Tensor) -> 'Geometry':
        return self.shifted(center - self.center)

    def rotated(self, angle):
        pivot = self.center
        return GeometryStack(math.map(lambda g: rotate(g, angle, pivot), self._geometries, dims=object), set_op=self._set_op)

    def __eq__(self, other):
        return isinstance(other, GeometryStack) \
               and self._shape == other._shape \
               and (self.geometries == other.geometries).all \
               and (self._set_op == other.set_op)

    def shallow_equals(self, other):
        if self is other:
            return True
        if not isinstance(other, GeometryStack) or self._geometries.shape != other.geometries.shape or self._set_op != other.set_op:
            return False
        return all(g1.shallow_equals(g2) for g1, g2 in zip(self.geometries, other.geometries))

    def __hash__(self):
        return hash(self._shape)

    def __getitem__(self, item):
        selected = self.geometries[slicing_dict(self, item)]
        if isinstance(selected, Geometry):
            return selected
        else:
            return GeometryStack(selected, set_op=self._set_op)

    @property
    def faces(self) -> 'Geometry':
        raise NotImplementedError

    @property
    def face_centers(self) -> Tensor:
        if self._set_op == 'intersection':
            raise NotImplementedError
        return math.map(lambda g: g.face_centers, self._geometries, dims=object)

    @property
    def face_areas(self) -> Tensor:
        if self._set_op == 'intersection':
            raise NotImplementedError
        return math.map(lambda g: g.face_areas, self._geometries, dims=object)

    @property
    def face_normals(self) -> Tensor:
        if self._set_op == 'intersection':
            raise NotImplementedError
        return math.map(lambda g: g.face_normals, self._geometries, dims=object)

    @property
    def boundary_elements(self) -> Dict[Any, Dict[str, slice]]:
        if self._set_op == 'intersection':
            raise NotImplementedError
        result = {}
        for idx in object_dims(self._geometries).meshgrid(names=True):
            elements = self._geometries[idx].boundary_elements
            for key, b_slice in elements.items():
                b_slice = {**idx ** b_slice}
                if key in result:
                    raise NotImplementedError(f"boundary slices {result} are not compatible with {b_slice}")
                else:
                    result[key] = b_slice
        return result

    @property
    def boundary_faces(self) -> Dict[Any, Dict[str, slice]]:
        raise NotImplementedError

    @property
    def face_shape(self) -> Shape:
        if self._set_op == 'intersection':
            raise NotImplementedError
        face_shapes = [g.face_shape for g in self.geometries]
        return shape_stack(self.geometries.shape, *face_shapes)

    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        raise NotImplementedError

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError

def _stack_geometries(geometries: Tuple[Geometry], set_op: str, dim=None) -> Geometry:
    assert set_op in ['union', 'intersection'], f"Set operation must be 'union' or 'intersection' but got {set_op}"
    if dim is None:
        dim = instance(set_op)
    assert dim.rank == 1 and dim.instance, f"{set_op} dimension must be a single instance dimension but got {dim}"
    if len(geometries) == 1 and isinstance(geometries[0], (tuple, list)):
        geometries = geometries[0]
    if len(geometries) == 0:
        return NO_GEOMETRY
    elif len(geometries) == 1:
        return geometries[0]
    elif set_op == 'union' and all(type(g) == type(geometries[0]) and isinstance(g, PhiTreeNode) for g in geometries):
        # ToDo look into using stacked attributes for intersection
        attrs = variable_attributes(geometries[0])
        values = {a: math.stack([getattr(g, a) for g in geometries], dim) for a in attrs}
        return copy_with(geometries[0], **values)
    else:
        # ToDo group by type, union individual types along union_<type>, then stack the groups
        base_geometries = ()
        for geometry in geometries:
            base_geometries += (geometry,)
        return math.stack(base_geometries, dim, set_op=set_op)

def union(*geometries, dim=instance('union')) -> Geometry:
    """
    Union of the given geometries.
    A point lies inside the union if it lies within at least one of the geometries.

    Args:
        *geometries: arbitrary geometries with same spatial dims. Arbitrary batch dims are allowed.
        dim: Union dimension. This must be an instance dimension.

    Returns:
        union `Geometry`
    """
    return _stack_geometries(geometries, 'union', dim)

def intersection(*geometries, dim=instance('intersection')) -> Geometry:
    """
    Intersection of the given geometries.
    A point lies inside the union if it lies within all of the geometries.

    Args:
        *geometries: arbitrary geometries with same spatial dims. Arbitrary batch dims are allowed.
        dim: Intersection dimension. This must be an instance dimension.

    Returns:
        intersection `Geometry`
    """
    return _stack_geometries(geometries, 'intersection', dim)


Geometry.__add__ = lambda g1, g2: union(g1, g2)
Geometry.__mul__ = lambda g1, g2: intersection(g1, g2)
Geometry.__or__ = lambda g1, g2: union(g1, g2)
Geometry.__and__ = lambda g1, g2: intersection(g1, g2)


def expel(geometry: Geometry,
          location: Tensor,
          min_separation: Union[float, Tensor] = 0,
          invert=False) -> Tensor:
    """
    Expels points at `location` out of the `geometry`.
    This implementation works with all geometries that implement `approximate_closest_surface()`.
    Specific geometries may override `Geometry.push()` for more accurate or efficient results.

    Args:
        geometry: `Geometry` that has an inside and outside.
        location: `Tensor` holding the positions before shifting.
        min_separation: Minimum distance between positions and surface after shifting.
        invert: Whether to invert the inside and outside of `geometry`.

    Returns:
        Tensor holding shifted positions.
    """
    if isinstance(geometry, InvertedGeometry):
        return expel(geometry.geometry, location, min_separation, invert=not invert)
    if isinstance(geometry, Box):  # legacy
        return geometry.push(location, not invert, min_separation)
    if math.always_close(geometry.bounding_radius(), 0):
        return location
    signed_distance, vec_to_surface, *_ = geometry.approximate_closest_surface(location)
    if not invert:
        shift_amount = math.maximum(0, min_separation - signed_distance)  # expel
        direction = math.safe_div(vec_to_surface, -signed_distance)  # this always points outward
    else:
        raise NotImplementedError
    return location + direction * shift_amount
