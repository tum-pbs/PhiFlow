import warnings
from typing import Dict, Tuple, Union, Optional, Any

import numpy as np

from phi import math
from phi.math import DimFilter
from phiml.math import rename_dims, vec, stack, expand, instance
from phiml.math._shape import parse_dim_order, dual, non_channel
from ._geom import Geometry, _keep_vector
from ..math import wrap, INF, Shape, channel, Tensor
from ..math.magic import slicing_dict


class BaseBox(Geometry):  # not a Subwoofer
    """
    Abstract base type for box-like geometries.
    """

    def __ne__(self, other):
        return not self == other

    @property
    def shape(self):
        raise NotImplementedError()

    @property
    def center(self) -> Tensor:
        raise NotImplementedError()

    @property
    def size(self) -> Tensor:
        raise NotImplementedError(self)

    @property
    def half_size(self) -> Tensor:
        raise NotImplementedError(self)

    @property
    def lower(self) -> Tensor:
        raise NotImplementedError(self)

    @property
    def upper(self) -> Tensor:
        raise NotImplementedError(self)

    @property
    def rotation_matrix(self) -> Optional[Tensor]:
        raise NotImplementedError(self)

    @property
    def is_size_variable(self):
        raise NotImplementedError(self)

    @property
    def volume(self) -> Tensor:
        return math.prod(self.size, 'vector')

    def bounding_radius(self):
        return math.vec_length(self.half_size)

    def global_to_local(self, global_position: Tensor, scale=True, origin='lower') -> Tensor:
        """
        Transform world-space coordinates into box-space coordinates.

        Args:
            global_position: World-space coordinates.
            scale: Whether to re-scale the output so that [0, 1] or [-1, 1] represent the box for `origin='lower'` or `origin='center'`, respectively.
            origin: 'lower' or 'center'

        Returns:
            Box-space coordinate `Tensor`
        """
        assert origin in ['lower', 'center', 'upper']
        origin_loc = getattr(self, origin)
        pos = global_position if math.always_close(origin_loc, 0) else global_position - origin_loc
        pos = math.rotate_vector(pos, self.rotation_matrix, invert=True)
        if scale:
            pos /= (self.half_size if origin == 'center' else self.size)
        return pos

    def local_to_global(self, local_position, scale=True, origin='lower'):
        assert origin in ['lower', 'center', 'upper']
        origin_loc = getattr(self, origin)
        pos = local_position * (self.half_size if origin == 'center' else self.size) if scale else local_position
        return math.rotate_vector(pos, self.rotation_matrix) + origin_loc

    def lies_inside(self, location):
        assert self.rotation_matrix is None, f"Please override lies_inside() for rotated boxes"
        bool_inside = (location >= self.lower) & (location <= self.upper)
        bool_inside = math.all(bool_inside, 'vector')
        bool_inside = math.any(bool_inside, self.shape.instance)  # union for instance dimensions
        return bool_inside

    def approximate_signed_distance(self, location: Union[Tensor, tuple]):
        """
        Computes the signed L-infinity norm (manhattan distance) from the location to the nearest side of the box.
        For an outside location `l` with the closest surface point `s`, the distance is `max(abs(l - s))`.
        For inside locations it is `-max(abs(l - s))`.

        Args:
          location: float tensor of shape (batch_size, ..., rank)

        Returns:
          float tensor of shape (*location.shape[:-1], 1).

        """
        location = self.global_to_local(location, scale=False, origin='center')
        distance = math.abs(location) - self.half_size
        distance = math.max(distance, 'vector')
        distance = math.min(distance, self.shape.instance)  # union for instance dimensions
        return distance

    def push(self, positions: Tensor, outward: bool = True, shift_amount: float = 0) -> Tensor:
        loc_to_center = self.global_to_local(positions, scale=False, origin='center')
        sgn_dist_from_surface = math.abs(loc_to_center) - self.half_size
        rotation_matrix = self.rotation_matrix
        if outward:
            # --- get negative distances (particles are inside) towards the nearest boundary and add shift_amount ---
            distances_of_interest = (sgn_dist_from_surface == math.max(sgn_dist_from_surface, 'vector')) & (sgn_dist_from_surface < 0)
            shift = distances_of_interest * (sgn_dist_from_surface - shift_amount)
            # ToDo reduce instance dim
        else:  # inward
            shift = (sgn_dist_from_surface + shift_amount) * (sgn_dist_from_surface > 0)  # get positive distances (particles are outside) and add shift_amount
            if instance(self):
                shift, loc_to_center, rotation_matrix = math.at_min((shift, loc_to_center, rotation_matrix), key=math.vec_length(shift), dim=instance)
            shift = math.where(abs(shift) > abs(loc_to_center), abs(loc_to_center), shift)  # ensure inward shift ends at center
        shift = math.rotate_vector(shift, rotation_matrix)
        return positions + math.where(loc_to_center < 0, 1, -1) * shift

    def approximate_closest_surface(self, location: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        loc_to_center = self.global_to_local(location, scale=False, origin='center')
        sgn_surf_delta = math.abs(loc_to_center) - self.half_size
        if instance(self):
            raise NotImplementedError
            self_center, self_radius, sgn_dist, center_delta, center_dist = math.at_min((self.center, self.radius, sgn_dist, center_delta, center_dist), key=abs(sgn_dist), dim=instance)
        # is_inside = math.all(sgn_surf_delta < 0, 'vector')
        # abs_surf_delta = abs(sgn_surf_delta)
        max_sgn_dist = math.max(sgn_surf_delta, 'vector')
        normal_axis = max_sgn_dist == sgn_surf_delta
        normal = math.vec_normalize(normal_axis * math.sign(loc_to_center))
        surf_to_center = math.where(normal_axis, math.sign(loc_to_center) * self.half_size, loc_to_center)
        closest_to_center = math.clip(surf_to_center, -self.half_size, self.half_size)
        surface_pos = self.local_to_global(closest_to_center, scale=False, origin='center')
        delta = surface_pos - location
        face_index = expand(0, non_channel(location))
        offset = normal.vector @ surface_pos.vector
        sgn_surf_dist = math.vec_length(delta) * math.sign(max_sgn_dist)
        return sgn_surf_dist, delta, normal, offset, face_index

    def project(self, *dimensions: str):
        """ Project this box into a lower-dimensional space. """
        warnings.warn("Box.project(dims) is deprecated. Use Box.vector[dims] instead", DeprecationWarning, stacklevel=2)
        return self.vector[dimensions]

    def sample_uniform(self, *shape: Shape) -> Tensor:
        uniform = math.random_uniform(self.shape.non_singleton.without('vector'), *shape, self.shape['vector'])
        return self.lower + uniform * self.size

    def sample_uniform_surface(self, *shape: Shape) -> Tensor:
        assert not instance(self), "sample_uniform_surface not yet supported for unions of boxes"
        samples = math.random_uniform(self.shape.non_singleton.non_instance, *shape, low=self.lower, high=self.upper)
        which = math.random_uniform(samples.shape.without('vector'))
        lo_or_up = math.where(which > .5, self.upper, self.lower)
        which = which * 2 % 1
        # --- which axis ---
        areas = self.face_areas
        total_area = math.sum(areas)
        frac_area = math.sum(areas / total_area, '~side')
        cum_area = math.cumulative_sum(frac_area, '~vector')
        axis = math.min(math.where(which <= cum_area, math.range(self.shape['vector'].as_dual()), self.spatial_rank), '~vector')
        axis_one_hot = math.scatter(math.zeros(samples.shape, dtype=bool), expand(axis, channel(index='vector')), True, treat_as_batch=samples.shape.without('vector'))
        math.assert_close(1, math.sum(axis_one_hot, 'vector'))
        samples = math.where(axis_one_hot, lo_or_up, samples)
        return samples

    def corner_representation(self) -> 'Box':
        assert self.rotation_matrix is None, f"corner_representation does not support rotations"
        return Box(self.lower, self.upper)

    box = corner_representation

    def center_representation(self, size_variable=True) -> 'Cuboid':
        return Cuboid(self.center, self.half_size, size_variable=size_variable)

    def contains(self, other: 'BaseBox'):
        """ Tests if the other box lies fully inside this box. """
        return np.all(other.lower >= self.lower) and np.all(other.upper <= self.upper)

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        return Cuboid(self.center, self.half_size * factor, size_variable=True)

    @property
    def boundary_elements(self) -> Dict[Any, Dict[str, slice]]:
        return {}

    @property
    def boundary_faces(self) -> Dict[Any, Dict[str, slice]]:
        return {}

    @property
    def faces(self) -> 'Geometry':
        return Cuboid(self.face_centers, self._half_size, self._rotation_matrix, size_variable=False)

    @property
    def face_centers(self) -> Tensor:
        return self.center + self.face_normals * self.half_size

    @property
    def face_normals(self) -> Tensor:
        unit_vectors = math.to_float(math.range(self.shape['vector']) == math.range(dual(**self.shape['vector'].untyped_dict)))
        vectors = math.rotate_vector(unit_vectors, self.rotation_matrix)
        return vectors * math.vec(dual('side'), lower=-1, upper=1)

    @property
    def face_areas(self) -> Tensor:
        others_mask = math.range(self.shape['vector']) != math.range(dual(**self.shape['vector'].untyped_dict))
        result = math.exp(math.sum(math.log(self.size) * others_mask, 'vector'))
        return expand(result, dual(side='lower,upper'))  # ~vector

    @property
    def face_shape(self) -> Shape:
        return self.shape.without('vector') & dual(side='lower,upper') & dual(**self.shape['vector'].untyped_dict)

    @property
    def corners(self):
        to_face = self.face_normals[{'~side': 'upper'}] * math.rename_dims(self.half_size, 'vector', dual)
        lower_upper = math.meshgrid(math.dual, **{dim: [-1, 1] for dim in self.vector.item_names}, stack_dim=dual('vector'))  # (x=2, y=2, ... vector=x,y,...)
        to_corner = math.sum(lower_upper * to_face, '~vector')
        return self.center + to_corner


class BoxType(type):
    """ Deprecated. Does not support item names. """

    def __getitem__(self, item):
        assert isinstance(item, tuple) and isinstance(item[0], str), "The Box constructor was updated in Φ-Flow version 2.2. Please add the dimension order as a comma-separated string as the first argument, e.g. Box['x,y', 0:1, 1:2] or use the kwargs constructor Box(x=1, y=(1, 2))"
        assert len(item) <= 4, f"Box[...] can only be used for x, y, z but got {len(item)} elements"
        dim_order = parse_dim_order(item[0])
        assert len(dim_order) == len(item) - 1, f"Dimension order '{item[0]}' does not match number of slices, {len(item) - 1}"
        lower = []
        upper = []
        for dim_name, dim in zip(dim_order, item[1:]):
            assert isinstance(dim, slice)
            assert dim.step is None or dim.step == 1, "Box: step must be 1 but is %s" % dim.step
            lower.append(dim.start if dim.start is not None else -np.inf)
            upper.append(dim.stop if dim.stop is not None else np.inf)
        vec = math.channel(vector=dim_order)
        lower = math.stack(lower, vec)
        upper = math.stack(upper, vec)
        return Box(lower, upper)


class Box(BaseBox, metaclass=BoxType):
    """
    Simple cuboid defined by location of lower and upper corner in physical space.

    Boxes can be constructed either from two positional vector arguments `(lower, upper)` or by specifying the limits by dimension name as `kwargs`.

    Examples:
        >>> Box(x=1, y=1)  # creates a two-dimensional unit box with `lower=(0, 0)` and `upper=(1, 1)`.
        >>> Box(x=(None, 1), y=(0, None)  # creates a Box with `lower=(-inf, 0)` and `upper=(1, inf)`.

        The slicing constructor was updated in version 2.2 and now requires the dimension order as the first argument.

        >>> Box['x,y', 0:1, 0:1]  # creates a two-dimensional unit box with `lower=(0, 0)` and `upper=(1, 1)`.
        >>> Box['x,y', :1, 0:]  # creates a Box with `lower=(-inf, 0)` and `upper=(1, inf)`.
    """

    def __init__(self, lower: Tensor = None, upper: Tensor = None, **size: Optional[Union[float, Tensor, tuple, list]]):
        """
        Args:
          lower: physical location of lower corner
          upper: physical location of upper corner
          **size: Specify size by dimension, either as `int` or `tuple` containing (lower, upper).
        """
        if lower is not None:
            assert isinstance(lower, Tensor), f"lower must be a Tensor but got {type(lower)}"
            assert 'vector' in lower.shape, "lower must have a vector dimension"
            assert lower.vector.item_names is not None, "vector dimension of lower must list spatial dimension order"
            self._lower = lower
        if upper is not None:
            assert isinstance(upper, Tensor), f"upper must be a Tensor but got {type(upper)}"
            assert 'vector' in upper.shape, "lower must have a vector dimension"
            assert upper.vector.item_names is not None, "vector dimension of lower must list spatial dimension order"
            self._upper = upper
        else:
            lower = []
            upper = []
            for item in size.values():
                if isinstance(item, (tuple, list)):
                    assert len(item) == 2, f"Box kwargs must be either dim=upper or dim=(lower,upper) but got {item}"
                    lo, up = item
                    lower.append(lo)
                    upper.append(up)
                elif item is None:
                    lower.append(-INF)
                    upper.append(INF)
                else:
                    lower.append(0)
                    upper.append(item)
            lower = [-INF if l is None else l for l in lower]
            upper = [INF if u is None else u for u in upper]
            self._upper = math.wrap(upper, math.channel(vector=tuple(size.keys())))
            self._lower = math.wrap(lower, math.channel(vector=tuple(size.keys())))
        vector_shape = self._lower.shape & self._upper.shape
        self._lower = math.expand(self._lower, vector_shape)
        self._upper = math.expand(self._upper, vector_shape)
        if self.size.vector.item_names is None:
            warnings.warn("Creating a Box without item names prevents certain operations like project()", DeprecationWarning, stacklevel=2)
        self._shape = self._lower.shape & self._upper.shape

    def __getitem__(self, item):
        item = _keep_vector(slicing_dict(self, item))
        return Box(self._lower[item], self._upper[item])

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **kwargs) -> 'Geometry':
        if all(isinstance(v, Box) for v in values):
            return NotImplemented  # stack attributes
        else:
            return Geometry.__stack__(values, dim, **kwargs)

    def without(self, dims: Tuple[str, ...]):
        remaining = list(self.shape.get_item_names('vector'))
        for dim in dims:
            if dim in remaining:
                remaining.remove(dim)
        return self.vector[remaining]

    def largest(self, dim: DimFilter) -> 'Box':
        dim = self.shape.without('vector').only(dim)
        if not dim:
            return self
        return Box(math.min(self._lower, dim), math.max(self._upper, dim))

    def __variable_attrs__(self):
        return '_lower', '_upper'

    def __value_attrs__(self):
        return '_lower', '_upper'

    @property
    def shape(self):
        if self._lower is None or self._upper is None:
            return self._shape
        return self._lower.shape & self._upper.shape

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @property
    def size(self):
        return self.upper - self.lower

    @property
    def center(self):
        return 0.5 * (self.lower + self.upper)

    @property
    def half_size(self):
        return self.size * 0.5

    @property
    def rotation_matrix(self) -> Optional[Tensor]:
        return None

    @property
    def is_size_variable(self):
        raise False

    def at(self, center: Tensor) -> 'BaseBox':
        return Cuboid(center, self.half_size, self.rotation_matrix)

    def shifted(self, delta, **delta_by_dim):
        return Box(self.lower + delta, self.upper + delta)

    def rotated(self, angle) -> Geometry:
        return self.center_representation().rotated(angle)

    def __mul__(self, other):
        if not isinstance(other, Box):
            return NotImplemented
        lower = self._lower.vector.unstack(self.spatial_rank) + other._lower.vector.unstack(other.spatial_rank)
        upper = self._upper.vector.unstack(self.spatial_rank) + other._upper.vector.unstack(other.spatial_rank)
        names = self._upper.vector.item_names + other._upper.vector.item_names
        lower = math.stack(lower, math.channel(vector=names))
        upper = math.stack(upper, math.channel(vector=names))
        return Box(lower, upper)

    def bounding_half_extent(self):
        return self.half_size

    def __repr__(self):
        if self._lower is None or self._upper is None:  # traced
            return f"Box[traced, shape={self._shape}]"
        if self.shape.non_channel.volume == 1:
            item_names = self.size.vector.item_names
            if item_names:
                return f"Box({', '.join([f'{dim}=({lo}, {up})' for dim, lo, up in zip(item_names, self._lower, self._upper)])})"
            else:  # deprecated
                return 'Box[%s at %s]' % ('x'.join([str(x) for x in self.size.numpy().flatten()]), ','.join([str(x) for x in self.lower.numpy().flatten()]))
        else:
            return f'Box[shape={self.shape}]'


class Cuboid(BaseBox):
    """Box specified by center position and half size."""

    def __init__(self,
                 center: Tensor = 0,
                 half_size: Union[float, Tensor] = None,
                 rotation: Optional[Tensor] = None,
                 size_variable=True,
                 **size: Union[float, Tensor]):
        """
        Args:
            center: Center position
            half_size: Half-size of the cuboid as vector or scalar
            rotation: Rotation angle(s) or rotation matrix.
            **size: Alternative way of specifying the size. If used, `half_size` must not be specified.
        """
        if half_size is not None:
            assert isinstance(half_size, Tensor), "half_size must be a Tensor"
            assert 'vector' in half_size.shape, f"Cuboid size must have a 'vector' dimension."
            assert half_size.shape.get_item_names('vector') is not None, f"Vector dimension must list spatial dimensions as item names. Use the syntax Cuboid(x=x, y=y) to assign names."
            self._half_size = half_size
        else:
            self._half_size = math.wrap(tuple(size.values()), math.channel(vector=tuple(size.keys()))) * 0.5
        center = wrap(center)
        if 'vector' not in center.shape or center.shape.get_item_names('vector') is None:
            center = math.expand(center, channel(self._half_size))
        self._center = center
        self._rotation_matrix = None if rotation is None else math.rotation_matrix(rotation)
        self._size_variable = size_variable

    def __repr__(self):
        return f"Cuboid(center={self._center}, half_size={self._half_size})"

    def __getitem__(self, item):
        item = _keep_vector(slicing_dict(self, item))
        rotation = self._rotation_matrix[item] if self._rotation_matrix is not None else None
        return Cuboid(self._center[item], self._half_size[item], rotation, size_variable=self._size_variable)

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **kwargs) -> 'Geometry':
        if all(isinstance(v, Cuboid) for v in values):
            size_variable = any([c._size_variable for c in values])
            if any(v._rotation_matrix is not None for v in values):
                matrices = [v._rotation_matrix for v in values]
                if any(m is None for m in matrices):
                    any_angle = math.rotation_angles([m for m in matrices if m is not None][0])
                    unit_matrix = math.rotation_matrix(any_angle * 0)
                    matrices = [unit_matrix if m is None else m for m in matrices]
                rotation = stack(matrices, dim, **kwargs)
            else:
                rotation = None
            return Cuboid(stack([v.center for v in values], dim, **kwargs), stack([v.half_size for v in values], dim, **kwargs), rotation, size_variable=size_variable)
        else:
            return Geometry.__stack__(values, dim, **kwargs)

    def __variable_attrs__(self):
        return ('_center', '_half_size', '_rotation_matrix') if self._size_variable else ('_center', '_rotation_matrix')

    def __value_attrs__(self):
        return '_center',

    @property
    def center(self):
        return self._center

    @property
    def half_size(self):
        return self._half_size

    @property
    def shape(self):
        if self._center is None or self._half_size is None:
            return None
        return self._center.shape & self._half_size.shape

    @property
    def size(self):
        return 2 * self._half_size

    @property
    def lower(self):
        return self._center - self._half_size

    @property
    def upper(self):
        return self._center + self._half_size

    @property
    def rotation_matrix(self) -> Optional[Tensor]:
        return self._rotation_matrix

    @property
    def is_size_variable(self):
        return self._size_variable

    def at(self, center: Tensor) -> 'BaseBox':
        return Cuboid(center, self.half_size, self.rotation_matrix, size_variable=self._size_variable)

    def rotated(self, angle) -> Geometry:
        if self._rotation_matrix is None:
            return Cuboid(self._center, self._half_size, angle, size_variable=self._size_variable)
        else:
            matrix = self._rotation_matrix @ (angle if dual(angle) else math.rotation_matrix(angle))
            return Cuboid(self._center, self._half_size, matrix, size_variable=self._size_variable)

    def bounding_half_extent(self):
        if self._rotation_matrix is not None:
            to_face = self.face_normals[{'~side': 0}] * math.rename_dims(self._half_size, 'vector', dual)
            return math.sum(abs(to_face), '~vector')
        return self.half_size

    def lies_inside(self, location):
        location = self.global_to_local(location, scale=False, origin='center')  # scale can only be performed for finite sizes
        bool_inside = abs(location) <= self._half_size
        bool_inside = math.all(bool_inside, 'vector')
        bool_inside = math.any(bool_inside, self.shape.instance)  # union for instance dimensions
        return bool_inside


def bounding_box(geometry):
    center = geometry.center
    extent = geometry.bounding_half_extent()
    return Box(lower=center - extent, upper=center + extent)
