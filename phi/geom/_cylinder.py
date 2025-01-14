from dataclasses import dataclass
from functools import cached_property
from typing import Union, Dict, Tuple, Optional, Sequence

from phiml import math
from phiml.math import (Shape, dual, wrap, Tensor, expand, vec, where, ncat, clip, length, normalize, minimum, vec_squared, channel, instance, stack, maximum, PI, linspace, sin, cos, sqrt, batch)
from phiml.math._magic_ops import all_attributes
from phiml.dataclasses import replace, sliceable
from ._geom import Geometry
from ._functions import rotate_vector as rotate, rotation_matrix, rotation_matrix_from_directions
from ._sphere import Sphere


@sliceable(keepdims='vector')
@dataclass(frozen=True)
class Cylinder(Geometry):
    """
    N-dimensional cylinder.
    Defined by center position, radius, depth, alignment axis, rotation.

    For cylinders whose bottom and top lie outside the domain or are otherwise not needed, you may use `infinite_cylinder` instead, which simplifies computations.
    """

    _center: Tensor
    radius: Tensor
    depth: Tensor
    rotation: Tensor  # rotation matrix
    axis: str

    variable_attrs: Tuple[str, ...] = ('_center', 'radius', 'depth', 'rotation')
    value_attrs: Tuple[str, ...] = ()

    @property
    def center(self) -> Tensor:
        return self._center

    @cached_property
    def shape(self) -> Shape:
        return self._center.shape & self.radius.shape & self.depth.shape & batch(self.rotation)

    @cached_property
    def radial_axes(self) -> Sequence[str]:
        return [d for d in self.shape.get_item_names('vector') if d != self.axis]

    @cached_property
    def volume(self) -> math.Tensor:
        return Sphere.volume_from_radius(self.radius, self.spatial_rank - 1) * self.depth

    @cached_property
    def up(self):
        return rotate(vec(**{d: 1 if d == self.axis else 0 for d in self.shape.get_item_names('vector')}), self._rot_or_none)

    @cached_property
    def rotation_matrix(self):
        return rotation_matrix(self.rotation, self.shape['vector'], none_to_unit=True)

    @property
    def _rot_or_none(self):
        return None if self.rotation is None else self.rotation_matrix

    def with_radius(self, radius: Tensor) -> 'Cylinder':
        return replace(self, radius=wrap(radius))

    def with_depth(self, depth: Tensor) -> 'Cylinder':
        return replace(self, depth=wrap(depth))

    def lies_inside(self, location):
        pos = rotate(location - self._center, self._rot_or_none, invert=True)
        r = pos.vector[self.radial_axes]
        h = pos.vector[self.axis]
        inside = (vec_squared(r) <= self.radius**2) & (h >= -.5*self.depth) & (h <= .5*self.depth)
        return math.any(inside, instance(self))  # union for instance dimensions

    def approximate_signed_distance(self, location: Union[Tensor, tuple]):
        location = rotate(location - self._center, self._rot_or_none, invert=True)
        r = location.vector[self.radial_axes]
        h = location.vector[self.axis]
        top_h = .5*self.depth
        # bot_h = -.5*self.depth
        # --- Compute distances ---
        radial_outward = normalize(r, 'vector', epsilon=1e-5)
        surf_r = radial_outward * self.radius
        radial_dist2 = vec_squared(r)
        inside_cyl = radial_dist2 <= self.radius**2
        clamped_r = where(inside_cyl, r, surf_r)
        # --- Closest point on bottom / top ---
        sgn_dist_side = abs(h) - top_h
        # --- Closest point on cylinder ---
        sgn_dist_cyl = length(r, 'vector') - self.radius
        # inside (all <= 0) -> largest SDF, outside (any > 0) -> largest positive SDF
        sgn_dist = maximum(sgn_dist_cyl, sgn_dist_side)
        return math.min(sgn_dist, instance(self))

    def approximate_closest_surface(self, location: Tensor):
        location = rotate(location - self._center, self._rot_or_none, invert=True)
        r = location.vector[self.radial_axes]
        h = location.vector[self.axis]
        top_h = .5*self.depth
        bot_h = -.5*self.depth
        # --- Compute distances ---
        radial_outward = normalize(r, 'vector', epsilon=1e-5)
        surf_r = radial_outward * self.radius
        radial_dist2 = vec_squared(r)
        inside_cyl = radial_dist2 <= self.radius**2
        clamped_r = where(inside_cyl, r, surf_r)
        # --- Closest point on bottom / top ---
        above = h >= 0
        flat_h = where(above, top_h, bot_h)
        on_flat = ncat([flat_h, clamped_r], self._center.shape['vector'])
        normal_flat = where(above, self.up, -self.up)
        # --- Closest point on cylinder ---
        clamped_h = clip(h, bot_h, top_h)
        on_cyl = ncat([surf_r, clamped_h], self._center.shape['vector'])
        normal_cyl = ncat([radial_outward, 0], self._center.shape['vector'], expand_values=True)
        # --- Choose closest ---
        d_flat = length(on_flat - location, 'vector')
        d_cyl = length(on_cyl - location, 'vector')
        flat_closer = d_flat <= d_cyl
        surf_point = where(flat_closer, on_flat, on_cyl)
        inside = inside_cyl & (h >= bot_h) & (h <= top_h)
        sgn_dist = minimum(d_flat, d_cyl) * where(inside, -1, 1)
        delta = surf_point - location
        normal = where(flat_closer, normal_flat, normal_cyl)
        delta = rotate(delta, self._rot_or_none)
        normal = rotate(normal, self._rot_or_none)
        idx = None
        if instance(self):
            sgn_dist, delta, normal, idx = math.min((sgn_dist, delta, normal, range), instance(self), key=sgn_dist)
        return sgn_dist, delta, normal, None, idx

    def sample_uniform(self, *shape: math.Shape):
        r = Sphere(self._center[self.radial_axes], self.radius).sample_uniform(*shape)
        h = math.random_uniform(*shape, -.5*self.depth, .5*self.depth)
        rh = ncat([r, h], self._center.shape['vector'])
        return rotate(rh, self._rot_or_none)

    def bounding_radius(self):
        return length(vec(rad=self.radius, dep=.5*self.depth), 'vector')

    def bounding_half_extent(self, epsilon=1e-5):
        if self.rotation is not None:
            tip = abs(self.up) * .5 * self.depth
            return tip + self.radius * sqrt(maximum(epsilon, 1 - self.up**2))
        return ncat([.5*self.depth, expand(self.radius, channel(vector=self.radial_axes))], self._center.shape['vector'], expand_values=True)

    def at(self, center: Tensor) -> 'Geometry':
        return replace(self, _center=center)

    def rotated(self, angle):
        rot = self.rotation_matrix @ rotation_matrix(angle) if self.rotation is not None else rotation_matrix(angle)
        return replace(self, rotation=rot)

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        return replace(self, radius=self.radius * factor, depth=self.depth * factor)

    @property
    def faces(self) -> 'Geometry':
        raise NotImplementedError(f"Cylinder.faces not implemented.")

    @property
    def face_centers(self) -> Tensor:
        fac = wrap([-.5, .5, 0], dual(shell='bottom,top,lateral'))
        return self.center + fac * self.depth * self.up

    @property
    def face_areas(self) -> Tensor:
        flat = Sphere.volume_from_radius(self.radius, self.spatial_rank - 1)
        lateral = 2*PI*self.radius * self.depth
        return stack({'bottom': flat, 'top': flat, 'lateral': lateral}, dual('shell'), expand_values=True)

    @property
    def face_normals(self) -> Tensor:
        raise NotImplementedError

    @property
    def boundary_elements(self) -> Dict[str, Tuple[Dict[str, slice], Dict[str, slice]]]:
        return {}

    @property
    def boundary_faces(self) -> Dict[str, Tuple[Dict[str, slice], Dict[str, slice]]]:
        return {}

    @property
    def face_shape(self) -> Shape:
        return self.shape.without('vector') & dual(shell='bottom,top,lateral')

    @property
    def corners(self) -> Tensor:
        return math.zeros(self.shape & dual(corners=0))

    def __eq__(self, other):
        return Geometry.__eq__(self, other)

    def vertex_rings(self, count: Shape) -> Tensor:
        if self.spatial_rank == 3:
            angle = linspace(0, 2*PI, count)
            h = stack({'bot': -.5 * self.depth, 'top': .5 * self.depth}, '~face')
            s = sin(angle) * self.radius
            c = cos(angle) * self.radius
            r = stack([s, c], channel(vector=self.radial_axes))
            x = ncat([h, r], self._center.shape['vector'], expand_values=True)
            return rotate(x, self._rot_or_none) + self._center
        raise NotImplementedError


def cylinder(center: Union[Tensor, float] = None,
             radius: Union[float, Tensor] = None,
             depth: Union[float, Tensor] = None,
             rotation: Optional[Tensor] = None,
             axis: Union[int, str, Tensor] = -1,
             variables=('center', 'radius', 'depth', 'rotation'),
             **center_: Union[float, Tensor]) -> Cylinder:
    """
    Args:
        center: Cylinder center as `Tensor` with `vector` dimension.
            The spatial dimension order should be specified in the `vector` dimension via item names.
            Can be left empty to specify dimensions via kwargs.
        radius: Cylinder radius as `float` or `Tensor`.
        depth: Cylinder length as `float` or `Tensor`.
        rotation: Rotation angle(s) or rotation matrix.
        axis: The cylinder is aligned along this axis, perturbed by `rotation`.
            Specified either as the dim along which the cylinder is aligned or as a vector.
        variables: Which properties of the cylinder are variable, i.e. traced and optimizable. All by default.
        **center_: Specifies center when the `center` argument is not given. Center position by dimension, e.g. `x=0.5, y=0.2`.
    """
    if center is not None:
        if not isinstance(center, Tensor):
            assert center == 0 and isinstance(axis, Tensor)
            center = expand(0, axis.shape['vector'])
        assert isinstance(center, Tensor), f"center must be a Tensor but got {type(center).__name__}"
        assert 'vector' in center.shape, f"Sphere center must have a 'vector' dimension."
        assert center.shape.get_item_names('vector') is not None, f"Vector dimension must list spatial dimensions as item names. Use the syntax Sphere(x=x, y=y) to assign names."
        center = center
    else:
        center = wrap(tuple(center_.values()), channel(vector=tuple(center_.keys())))
    assert radius is not None, "radius must be specified"
    radius = wrap(radius)
    if depth is None:
        assert isinstance(axis, Tensor)
        depth = 2 * length(axis, 'vector')
    else:
        depth = wrap(depth)
    axis = center.vector.item_names[axis] if isinstance(axis, int) else axis
    if isinstance(axis, Tensor):  # specify cylinder axis as vector
        assert 'vector' in axis.shape, f"When specifying axis a Tensor, it must have a 'vector' dimension."
        assert rotation is None, f"When specifying axis as a "
        axis_ = center.vector.item_names[-1]
        unit_vec = vec(**{d: 1 if d == axis_ else 0 for d in center.vector.item_names})
        rotation = rotation_matrix_from_directions(unit_vec, axis, epsilon=1e-5)
        axis = axis_
    else:
        rotation = rotation_matrix(rotation)
    variables = [{'center': '_center'}.get(v, v) for v in variables]
    assert 'vector' not in radius.shape, f"Cylinder radius must not vary along vector but got {radius}"
    assert set(variables).issubset(set(all_attributes(Cylinder))), f"Invalid variables: {variables}"
    assert axis in center.vector.item_names, f"Cylinder axis {axis} not part of vector dim {center.vector}"
    return Cylinder(center, radius, depth, rotation, axis, tuple(variables), ())
