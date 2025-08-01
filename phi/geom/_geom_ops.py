import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import Union, Dict, Any, Tuple, Sequence

from phi import math
from phiml import math, stack
from phiml.dataclasses import sliceable
from phiml.math import wrap, merge_shapes
from phiml.math._shape import shape_stack, Shape, EMPTY_SHAPE, channel
from phiml.math._tensors import object_dims, layout
from phiml.math.magic import PhiTreeNode
from ._box import bounding_box, Box
from ._functions import vec_length
from ._geom import Geometry, NoGeometry
from ._geom import InvertedGeometry
from ._transform import rotate
from ..math import Tensor, instance
from ..math.magic import slicing_dict


@sliceable(keepdims='vector')
@dataclass(frozen=True, eq=False)
class GeometryStack(Geometry):
    """
    Represents a tensor of Geometries.
    Instance dimensions represent geometry unions and are reduced.
    """
    geometries: Tensor
    
    def __post_init__(self):
        assert isinstance(self.geometries, Tensor) and self.geometries.dtype.kind == object, f"geometries must be a Tensor of geometries but got {self.geometries}"
        ranks = wrap(math.map(lambda g: g.spatial_rank, self.geometries, dims=object))
        assert ranks.min == ranks.max, f"Can only stack geometries of the same spatial rank but got ranks {ranks}"

    @cached_property
    def shape(self) -> Shape:
        return self.geometries.shape

    @property
    def sets(self) -> Dict[str, Shape]:
        all_names = set()
        for g in self.geometries:
            all_names.update(g.sets)
        result = {}
        for name in all_names:
            all_dims = []
            for g in self.geometries:
                all_dims.append(g.sets.get(name, EMPTY_SHAPE))
            all_dims = merge_shapes(all_dims, allow_varying_sizes=True)
            zeros = all_dims.with_sizes(0)
            set_shapes = []
            for g in self.geometries:
                set_shapes.append(g.sets.get(name, zeros))
            set_shape = shape_stack(object_dims(self.geometries), *set_shapes)
            result[name] = set_shape
        return result

    @property
    def object_dims(self):
        return object_dims(self.geometries)

    def unstack(self, dimension) -> tuple:
        if dimension == self.geometries.shape.name:
            return tuple(self.geometries)
        raise NotImplementedError()

    def _bounding_box(self):
        boxes = math.map(bounding_box, self.geometries, dims=object)
        return boxes.largest(instance)

    @property
    def center(self) -> Tensor:
        return self._bounding_box().center

    @property
    def weighted_center(self):
        centers = math.map(lambda g: g.center, self.geometries, dims=object)
        if not instance(centers):
            return centers
        # --- volume-weighted mean over instance dimensions ---
        vol = math.map(lambda g: g.volume, self.geometries, dims=object)
        return math.sum(centers * vol, instance) / math.sum(vol, instance)  # ToDo could also return bounding box center

    @property
    def spatial_rank(self) -> int:
        return next(iter(self.geometries)).spatial_rank

    @property
    def volume(self) -> math.Tensor:
        if instance(self.geometries):
            warnings.warn("Volume of a union assumes geometries do not overlap and may not be accurate otherwise.", RuntimeWarning)
        vol = math.map(lambda g: g.volume, self.geometries, dims=object)
        return math.sum(vol, instance(self.geometries))

    def lies_inside(self, location: math.Tensor):
        inside = math.map(lambda g, l: g.lies_inside(l), self.geometries, location, dims=object)
        return math.any(inside, instance(self.geometries))

    def approximate_signed_distance(self, location: math.Tensor):
        dist = math.map(lambda g, l: g.approximate_signed_distance(l), self.geometries, location, dims=object)
        return math.min(dist, instance(self.geometries))

    def approximate_closest_surface(self, location: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        signed_dist, delta, normals, offsets, face_idx = math.map(lambda g, l: g.approximate_closest_surface(l), self.geometries, location, dims=object)
        return math.at_min((signed_dist, delta, normals, offsets, face_idx), key=abs(signed_dist), dim=instance)

    def bounding_radius(self):
        center = self.center
        rad = math.map(lambda g: vec_length(g.center - center) + g.bounding_radius(), self.geometries, dims=object)
        return math.max(rad, instance(self.geometries))

    def bounding_half_extent(self):
        return self._bounding_box().half_size

    def shifted(self, delta: Tensor) -> 'Geometry':
        return GeometryStack(math.map(lambda g: g.shifted(delta), self.geometries, dims=object))

    def at(self, center: Tensor) -> 'Geometry':
        return self.shifted(center - self.center)

    def rotated(self, angle):
        pivot = self.center
        return GeometryStack(math.map(lambda g: rotate(g, angle, pivot), self.geometries, dims=object))

    def __getitem__(self, item):
        selected = self.geometries[slicing_dict(self, item)]
        if isinstance(selected, Geometry):
            return selected
        else:
            return GeometryStack(selected)

    @property
    def faces(self) -> 'Geometry':
        raise NotImplementedError

    @property
    def face_centers(self) -> Tensor:
        return math.map(lambda g: g.face_centers, self.geometries, dims=object)

    @property
    def face_areas(self) -> Tensor:
        return math.map(lambda g: g.face_areas, self.geometries, dims=object)

    @property
    def face_normals(self) -> Tensor:
        return math.map(lambda g: g.face_normals, self.geometries, dims=object)

    @property
    def boundary_elements(self) -> Dict[Any, Dict[str, slice]]:
        result = {}
        for idx in object_dims(self.geometries).meshgrid(names=True):
            elements = self.geometries[idx].boundary_elements
            for key, b_slice in elements.items():
                b_slice = {**idx **b_slice}
                if key in result:
                    raise NotImplementedError(f"boundary slices {result} are not compatible with {b_slice}")
                else:
                    result[key] = b_slice
        return result

    @property
    def boundary_faces(self) -> Dict[str, Dict[str, slice]]:
        return next(iter(self.geometries)).boundary_faces

    @property
    def face_shape(self) -> Shape:
        face_shapes = [g.face_shape for g in self.geometries]
        return shape_stack(self.geometries.shape, *face_shapes)

    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        raise NotImplementedError

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError


@sliceable(keepdims='vector')
@dataclass(frozen=True, eq=False)
class Intersection(Geometry):
    geometries: Tensor

    def __post_init__(self):
        assert isinstance(self.geometries, Tensor) and self.geometries.dtype.kind == object, f"geometries must be a Tensor of geometries but got {self.geometries}"
        ranks = wrap(math.map(lambda g: g.spatial_rank, self.geometries, dims=object))
        assert ranks.min == ranks.max, f"Can only stack geometries of the same spatial rank but got ranks {ranks}"

    @cached_property
    def shape(self) -> Shape:
        return self.geometries.shape

    def _bounding_box(self):
        boxes = math.map(bounding_box, self.geometries, dims=object)
        return boxes.smallest(instance)

    @property
    def center(self) -> Tensor:
        centers = math.map(lambda g: g.center, self.geometries, dims=object)
        if not instance(centers):
            return centers
        # --- bounding box of smallest volume ---
        warnings.warn("Center of an intersection assumes geometries are subsets of each larger geometry and may not be accurate otherwise.", RuntimeWarning)
        vol = math.map(lambda g: g.volume, self.geometries, dims=object)
        boxes = math.map(bounding_box, self.geometries, dims=object)
        smallest_vol_idx = math.argmin(vol, instance)
        return boxes[smallest_vol_idx].center

    @property
    def volume(self) -> math.Tensor:
        if instance(self.geometries):
            warnings.warn("Volume of an intersection assumes geometries are subsets of each larger geometry and may not be accurate otherwise.", RuntimeWarning)
        vol = math.map(lambda g: g.volume, self.geometries, dims=object)
        return math.min(vol, instance(self.geometries))

    def lies_inside(self, location: math.Tensor):
        inside = math.map(lambda g, l: g.lies_inside(l), self.geometries, location, dims=object)
        return math.all(inside, instance(self.geometries))

    def approximate_signed_distance(self, location: math.Tensor):
        dist = math.map(lambda g, l: g.approximate_signed_distance(l), self.geometries, location, dims=object)
        return math.max(dist, instance(self.geometries))

    def approximate_fraction_inside(self, other_geometry: Geometry, balance: Union[Tensor, float] = 0.5) -> Tensor:
        assert isinstance(other_geometry, Geometry)
        radius = other_geometry.bounding_radius()
        location = other_geometry.center
        distances = math.map(lambda g: g.approximate_signed_distance(location), self.geometries, dims=object)
        inside_fraction = balance - distances / radius
        inside_fraction = math.clip(inside_fraction, 0, 1)
        inside_fraction = math.min(inside_fraction, instance(self.geometries))
        return inside_fraction

    def approximate_closest_surface(self, location: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError

    def bounding_radius(self):
        raise NotImplementedError

    def bounding_half_extent(self):
        return self._bounding_box().half_size

    def shifted(self, delta: Tensor) -> 'Geometry':
        return Intersection(math.map(lambda g: g.shifted(delta), self.geometries, dims=object))

    def at(self, center: Tensor) -> 'Geometry':
        return self.shifted(center - self.center)

    def rotated(self, angle):
        pivot = self.center
        return Intersection(math.map(lambda g: rotate(g, angle, pivot), self.geometries, dims=object))

    def __getitem__(self, item):
        selected = self.geometries[slicing_dict(self, item)]
        if isinstance(selected, Geometry):
            return selected
        else:
            return GeometryStack(selected)

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
    def boundary_elements(self) -> Dict[Any, Dict[str, slice]]:
        for g in self.geometries:
            if g.boundary_elements:
                raise NotImplementedError
        return {}

    @property
    def boundary_faces(self) -> Dict[str, Dict[str, slice]]:
        return next(iter(self.geometries)).boundary_faces

    @property
    def face_shape(self) -> Shape:
        raise NotImplementedError

    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        raise NotImplementedError

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError


def union(*geometries, dim=instance('union')):
    """
    Union of the given geometries.
    A point lies inside the union if it lies within at least one of the geometries.

    Args:
        *geometries: arbitrary geometries with same spatial dims. Arbitrary batch dims are allowed.
        dim: Union dimension. This must be an instance dimension.

    Returns:
        union `Geometry`
    """
    assert dim.rank == 1 and dim.instance, f"union dimension must be a single instance dimension but got {dim}"
    geometries = geometries[0] if len(geometries) == 1 and isinstance(geometries[0], (tuple, list)) else geometries
    if len(geometries) == 0:
        warnings.warn("Empty union cannot infer dimensionality. Returning 0-dimensional empty.", RuntimeWarning, stacklevel=2)
        return NoGeometry(channel(vector=0))
    elif len(geometries) == 1:
        return geometries[0]
    elif all(type(g) == type(geometries[0]) and isinstance(g, PhiTreeNode) for g in geometries):
        return stack(tuple(geometries), dim, simplify=True)
    else:
        return GeometryStack(layout(geometries, dim))


def intersection(*geometries: Geometry, dim=instance('intersection')) -> Geometry:
    """
    Intersection of the given geometries.
    A point lies inside the union if it lies within all of the geometries.

    Args:
        *geometries: arbitrary geometries with same spatial dims. Arbitrary batch dims are allowed.
        dim: Intersection dimension. This must be an instance dimension.

    Returns:
        intersection `Geometry`
    """
    assert dim.rank == 1 and dim.instance, f"intersection dimension must be a single instance dimension but got {dim}"
    geometries = geometries[0] if len(geometries) == 1 and isinstance(geometries[0], (tuple, list)) else geometries
    if len(geometries) == 0:
        warnings.warn("Empty intersection cannot infer dimensionality. Returning 0-dimensional empty.", RuntimeWarning, stacklevel=2)
        return NoGeometry(channel(vector=0))
    elif len(geometries) == 1:
        return geometries[0]
    return Intersection(layout(geometries, dim))


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
