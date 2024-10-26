from dataclasses import dataclass
from functools import cached_property
from typing import Union, Dict, Tuple, Optional, Sequence

from phiml import math
from phiml.math import Shape, dual, wrap, Tensor, expand, vec, where, ccat, clip, length, normalize, rotate_vector, minimum, vec_squared, rotation_matrix, channel, instance, stack, maximum, PI, linspace, sin, cos, \
    rotation_matrix_from_directions
from phiml.math._magic_ops import all_attributes
from phiml.math.magic import slicing_dict
from ._geom import Geometry, _keep_vector
from ._sphere import Sphere


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
        return self._center.shape & self.radius.shape & self.depth.shape

    @cached_property
    def radial_axes(self) -> Sequence[str]:
        return [d for d in self._center.vector.item_names if d != self.axis]

    @cached_property
    def volume(self) -> math.Tensor:
        return Sphere.volume_from_radius(self.radius, self.spatial_rank - 1) * self.depth

    @cached_property
    def up(self):
        return math.rotate_vector(vec(**{d: 1 if d == self.axis else 0 for d in self._center.vector.item_names}), self.rotation)

    def lies_inside(self, location):
        pos = rotate_vector(location - self._center, self.rotation, invert=True)
        r = pos.vector[self.radial_axes]
        h = pos.vector[self.axis]
        inside = (vec_squared(r) <= self.radius**2) & (h >= -.5*self.depth) & (h <= .5*self.depth)
        return math.any(inside, instance(self))  # union for instance dimensions

    def approximate_signed_distance(self, location: Union[Tensor, tuple]):
        location = math.rotate_vector(location - self._center, self.rotation, invert=True)
        r = location.vector[self.radial_axes]
        h = location.vector[self.axis]
        top_h = .5*self.depth
        bot_h = -.5*self.depth
        # --- Compute distances ---
        radial_outward = normalize(r, epsilon=1e-5)
        surf_r = radial_outward * self.radius
        radial_dist2 = vec_squared(r)
        inside_cyl = radial_dist2 <= self.radius**2
        clamped_r = where(inside_cyl, r, surf_r)
        # --- Closest point on bottom / top ---
        sgn_dist_side = abs(h) - top_h
        # --- Closest point on cylinder ---
        sgn_dist_cyl = length(r) - self.radius
        # inside (all <= 0) -> largest SDF, outside (any > 0) -> largest positive SDF
        sgn_dist = maximum(sgn_dist_cyl, sgn_dist_side)
        return math.min(sgn_dist, instance(self))

    def approximate_closest_surface(self, location: Tensor):
        location = math.rotate_vector(location - self._center, self.rotation, invert=True)
        r = location.vector[self.radial_axes]
        h = location.vector[self.axis]
        top_h = .5*self.depth
        bot_h = -.5*self.depth
        # --- Compute distances ---
        radial_outward = normalize(r, epsilon=1e-5)
        surf_r = radial_outward * self.radius
        radial_dist2 = vec_squared(r)
        inside_cyl = radial_dist2 <= self.radius**2
        clamped_r = where(inside_cyl, r, surf_r)
        # --- Closest point on bottom / top ---
        above = h >= 0
        flat_h = where(above, top_h, bot_h)
        on_flat = ccat([flat_h, clamped_r], self._center.shape['vector'])
        normal_flat = where(above, self.up, -self.up)
        # --- Closest point on cylinder ---
        clamped_h = clip(h, bot_h, top_h)
        on_cyl = ccat([surf_r, clamped_h], self._center.shape['vector'])
        normal_cyl = ccat([radial_outward, 0], self._center.shape['vector'], expand_values=True)
        # --- Choose closest ---
        d_flat = length(on_flat - location)
        d_cyl = length(on_cyl - location)
        flat_closer = d_flat <= d_cyl
        surf_point = where(flat_closer, on_flat, on_cyl)
        inside = inside_cyl & (h >= bot_h) & (h <= top_h)
        sgn_dist = minimum(d_flat, d_cyl) * where(inside, -1, 1)
        delta = surf_point - location
        normal = where(flat_closer, normal_flat, normal_cyl)
        delta = rotate_vector(delta, self.rotation)
        normal = rotate_vector(normal, self.rotation)
        if instance(self):
            sgn_dist, delta, normal = math.at_min((sgn_dist, delta, normal), key=sgn_dist, dim=instance(self))
        return sgn_dist, delta, normal, None, None

    def sample_uniform(self, *shape: math.Shape):
        r = Sphere(self._center[self.radial_axes], self.radius).sample_uniform(*shape)
        h = math.random_uniform(*shape, -.5*self.depth, .5*self.depth)
        rh = ccat([r, h], self._center.shape['vector'])
        return rotate_vector(rh, self.rotation)

    def bounding_radius(self):
        return math.length(vec(rad=self.radius, dep=.5*self.depth))

    def bounding_half_extent(self):
        if self.rotation is not None:
            return expand(self.bounding_radius(), self._center.shape.only('vector'))
        return ccat([.5*self.depth, expand(self.radius, channel(vector=self.radial_axes))], self._center.shape['vector'])

    def at(self, center: Tensor) -> 'Geometry':
        return Cylinder(center, self.radius, self.depth, self.rotation, self.axis, self.variable_attrs, self.value_attrs)

    def rotated(self, angle):
        rot = self.rotation @ rotation_matrix(angle) if self.rotation is not None else rotation_matrix(angle)
        return Cylinder(self.center, self.radius, self.depth, rot, self.axis, self.variable_attrs, self.value_attrs)

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        return Cylinder(self._center, self.radius * factor, self.depth * factor, self.rotation, self.axis, self.variable_attrs, self.value_attrs)

    def __getitem__(self, item):
        item = slicing_dict(self, item)
        return Cylinder(self._center[_keep_vector(item)], self.radius[item], self.depth[item], math.slice(self.rotation, item), self.axis, self.variable_attrs, self.value_attrs)

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **kwargs) -> 'Geometry':
        if all(isinstance(v, Cylinder) for v in values) and all(v.axis == values[0].axis for v in values):
            var_attrs = set()
            var_attrs.update(*[set(v.variable_attrs) for v in values])
            val_attrs = set()
            val_attrs.update(*[set(v.value_attrs) for v in values])
            if any(v.rotation is not None for v in values):
                matrices = [v.rotation for v in values]
                if any(m is None for m in matrices):
                    any_angle = math.rotation_angles([m for m in matrices if m is not None][0])
                    unit_matrix = math.rotation_matrix(any_angle * 0)
                    matrices = [unit_matrix if m is None else m for m in matrices]
                rotation = stack(matrices, dim, **kwargs)
            else:
                rotation = None
            center = stack([v.center for v in values], dim, simplify=True, **kwargs)
            radius = stack([v.radius for v in values], dim, simplify=True, **kwargs)
            depth = stack([v.depth for v in values], dim, simplify=True, **kwargs)
            return Cylinder(center, radius, depth, rotation, values[0].axis, tuple(var_attrs), tuple(val_attrs))
        else:
            return Geometry.__stack__(values, dim, **kwargs)

    @property
    def faces(self) -> 'Geometry':
        raise NotImplementedError(f"Cylinder.faces not implemented.")

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
            x = ccat([h, r], self._center.shape['vector'], expand_values=True)
            return math.rotate_vector(x, self.rotation) + self._center
        raise NotImplementedError


def cylinder(center: Tensor = None,
             radius: Union[float, Tensor] = None,
             depth: Union[float, Tensor] = None,
             rotation: Optional[Tensor] = None,
             axis: int | str | Tensor = -1,
             variables=('center', 'radius', 'depth', 'rotation'),
             **center_: Union[float, Tensor]):
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
        assert isinstance(center, Tensor), f"center must be a Tensor but got {type(center).__name__}"
        assert 'vector' in center.shape, f"Sphere center must have a 'vector' dimension."
        assert center.shape.get_item_names('vector') is not None, f"Vector dimension must list spatial dimensions as item names. Use the syntax Sphere(x=x, y=y) to assign names."
        center = center
    else:
        center = wrap(tuple(center_.value_attrs()), channel(vector=tuple(center_.keys())))
    radius = wrap(radius)
    depth = wrap(depth)
    axis = center.vector.item_names[axis] if isinstance(axis, int) else axis
    if isinstance(axis, Tensor):  # specify cylinder axis as vector
        assert 'vector' in axis.shape, f"When specifying axis a Tensor, it must have a 'vector' dimension."
        assert rotation is None, f"When specifying axis as a "
        axis_ = center.vector.item_names[-1]
        unit_vec = vec(**{d: 1 if d == axis_ else 0 for d in center.vector.item_names})
        rotation = rotation_matrix_from_directions(unit_vec, axis)
        axis = axis_
    else:
        rotation = rotation_matrix(rotation)
    variables = [{'center': '_center'}.get(v, v) for v in variables]
    assert 'vector' not in radius.shape, f"Cylinder radius must not vary along vector but got {radius}"
    assert set(variables).issubset(set(all_attributes(Cylinder))), f"Invalid variables: {variables}"
    assert axis in center.vector.item_names, f"Cylinder axis {axis} not part of vector dim {center.vector}"
    return Cylinder(center, radius, depth, rotation, axis, tuple(variables), ())
