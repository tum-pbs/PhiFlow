from typing import Union, Dict, Tuple, Optional, Sequence

from phiml import math
from phiml.math import Shape, dual, wrap, Tensor, expand, vec, where, ccat, clip, length, normalize, rotate_vector, minimum, vec_squared, rotation_matrix, channel, instance, stack, \
    maximum
from phiml.math.magic import slicing_dict
from ._geom import Geometry, _keep_vector
from ._sphere import Sphere


class Cylinder(Geometry):
    """
    N-dimensional cylinder.
    Defined by center position, radius, depth, alignment axis, rotation.

    For cylinders whose bottom and top lie outside the domain or are otherwise not needed, you may use `infinite_cylinder` instead, which simplifies computations.
    """

    def __init__(self,
                 center: Tensor = None,
                 radius: Union[float, Tensor] = None,
                 depth: Union[float, Tensor] = None,
                 rotation: Optional[Tensor] = None,
                 axis=-1,
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
            variables: Which properties of the cylinder are variable, i.e. traced and optimizable. All by default.
            **center_: Specifies center when the `center` argument is not given. Center position by dimension, e.g. `x=0.5, y=0.2`.
        """
        if center is not None:
            assert isinstance(center, Tensor), f"center must be a Tensor but got {type(center).__name__}"
            assert 'vector' in center.shape, f"Sphere center must have a 'vector' dimension."
            assert center.shape.get_item_names('vector') is not None, f"Vector dimension must list spatial dimensions as item names. Use the syntax Sphere(x=x, y=y) to assign names."
            self._center = center
        else:
            self._center = wrap(tuple(center_.values()), channel(vector=tuple(center_.keys())))
        self._radius = wrap(radius)
        self._depth = wrap(depth)
        self._rotation = None if rotation is None else rotation_matrix(rotation)
        self._variables = tuple([v if v.startswith('_') else '_' + v for v in variables])
        self._axis = self._center.vector.item_names[axis] if isinstance(axis, int) else axis
        assert 'vector' not in self._radius.shape, f"Cylinder radius must not vary along vector but got {radius}"
        assert set(self._variables).issubset(set(self.__all_attrs__())), f"Invalid variables: {self._variables}"
        assert self._axis in self._center.vector.item_names, f"Cylinder axis {self._axis} not part of vector dim {self._center.vector}"

    def __all_attrs__(self) -> tuple:
        return '_center', '_radius', '_depth', '_rotation'

    def __variable_attrs__(self) -> tuple:
        return self._variables

    def __value_attrs__(self) -> tuple:
        return ()

    @property
    def shape(self) -> Shape:
        if self._center is None or self._radius is None or self._depth is None:
            raise RuntimeError
        return self._center.shape & self._radius.shape & self._depth.shape

    @property
    def radius(self) -> Tensor:
        return self._radius

    @property
    def center(self) -> Tensor:
        return self._center

    @property
    def depth(self) -> Tensor:
        return self._depth

    @property
    def axis(self) -> str:
        return self._axis

    @property
    def radial_axes(self) -> Sequence[str]:
        return [d for d in self._center.vector.item_names if d != self._axis]

    @property
    def rotation_matrix(self):
        return self._rotation

    @property
    def volume(self) -> math.Tensor:
        return Sphere.volume_from_radius(self._radius, self.spatial_rank - 1) * self._depth

    @property
    def up(self):
        return math.rotate_vector(vec(**{d: 1 if d == self._axis else 0 for d in self._center.vector.item_names}), self._rotation)

    def lies_inside(self, location):
        pos = rotate_vector(location - self._center, self._rotation, invert=True)
        r = pos.vector[self.radial_axes]
        h = pos.vector[self._axis]
        inside = (vec_squared(r) <= self._radius**2) & (h >= -.5*self._depth) & (h <= .5*self._depth)
        return math.any(inside, instance(self))  # union for instance dimensions

    def approximate_signed_distance(self, location: Union[Tensor, tuple]):
        location = math.rotate_vector(location - self._center, self._rotation, invert=True)
        r = location.vector[self.radial_axes]
        h = location.vector[self._axis]
        top_h = .5*self._depth
        bot_h = -.5*self._depth
        # --- Compute distances ---
        radial_outward = normalize(r, epsilon=1e-5)
        surf_r = radial_outward * self._radius
        radial_dist2 = vec_squared(r)
        inside_cyl = radial_dist2 <= self._radius**2
        clamped_r = where(inside_cyl, r, surf_r)
        # --- Closest point on bottom / top ---
        sgn_dist_side = abs(h) - top_h
        # --- Closest point on cylinder ---
        sgn_dist_cyl = length(r) - self._radius
        # inside (all <= 0) -> largest SDF, outside (any > 0) -> largest positive SDF
        sgn_dist = maximum(sgn_dist_cyl, sgn_dist_side)
        return math.min(sgn_dist, instance(self))

    def approximate_closest_surface(self, location: Tensor):
        location = math.rotate_vector(location - self._center, self._rotation, invert=True)
        r = location.vector[self.radial_axes]
        h = location.vector[self._axis]
        top_h = .5*self._depth
        bot_h = -.5*self._depth
        # --- Compute distances ---
        radial_outward = normalize(r, epsilon=1e-5)
        surf_r = radial_outward * self._radius
        radial_dist2 = vec_squared(r)
        inside_cyl = radial_dist2 <= self._radius**2
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
        delta = rotate_vector(delta, self._rotation)
        normal = rotate_vector(normal, self._rotation)
        if instance(self):
            sgn_dist, delta, normal = math.at_min((sgn_dist, delta, normal), key=sgn_dist, dim=instance(self))
        return sgn_dist, delta, normal, None, None

    def sample_uniform(self, *shape: math.Shape):
        raise NotImplementedError

    def bounding_radius(self):
        return math.length(vec(rad=self._radius, dep=.5*self._depth))

    def bounding_half_extent(self):
        if self._rotation is not None:
            return expand(self.bounding_radius(), self._center.shape.only('vector'))
        return ccat([.5*self._depth, expand(self._radius, channel(vector=self.radial_axes))], self._center.shape['vector'])

    def at(self, center: Tensor) -> 'Geometry':
        return Cylinder(center, self._radius, self._depth, self._rotation, self._axis, self._variables)

    def rotated(self, angle):
        if self._rotation is None:
            return Cylinder(self._center, self._radius, self._depth, angle, self._axis, self._variables)
        else:
            matrix = self._rotation @ (angle if dual(angle) else math.rotation_matrix(angle))
            return Cylinder(self._center, self._radius, self._depth, matrix, self._axis, self._variables)

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        return Cylinder(self._center, self._radius * factor, self._depth * factor, self._rotation, self._axis, self._variables)

    def __getitem__(self, item):
        item = slicing_dict(self, item)
        return Cylinder(self._center[_keep_vector(item)], self._radius[item], self._depth[item], math.slice(self._rotation, item), self._axis, self._variables)

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **kwargs) -> 'Geometry':
        if all(isinstance(v, Cylinder) for v in values) and all(v._axis == values[0]._axis for v in values):
            variables = set()
            variables.update(*[set(v._variables) for v in values])
            if any(v._rotation is not None for v in values):
                matrices = [v._rotation for v in values]
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
            return Cylinder(center, radius, depth, rotation, values[0]._axis, variables)
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
