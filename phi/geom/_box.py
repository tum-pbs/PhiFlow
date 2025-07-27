import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Tuple, Union, Optional, Any

import numpy as np

from phi import math
from phi.math import DimFilter
from phiml import ccat, stack, where, is_finite, concat
from phiml.dataclasses import sliceable, replace
from phiml.math import expand, instance, to_float, is_none
from phiml.math._shape import parse_dim_order, dual, non_channel, non_batch, shape
from . import rotate, rotation_matrix
from ._functions import vec_length
from ._geom import Geometry
from ..math import wrap, INF, Shape, channel, Tensor


class BoxType(type):
    """ Deprecated. Does not support item names. """

    def __call__(cls, *args, **kwargs):
        if 'lower' in kwargs or 'upper' in kwargs or (not kwargs and len(args) == 2) or (not args and 'pos' not in kwargs):
            return box_from_limits(*args, **kwargs)
        return type.__call__(cls, *args, **kwargs)

    def __getitem__(cls, item):
        assert isinstance(item, tuple) and isinstance(item[0], str), "The Box constructor was updated in Φ-Flow version 2.2. Please add the dimension order as a comma-separated string as the first argument, e.g. Box['x,y', 0:1, 1:2] or use the kwargs constructor Box(x=1, y=(1, 2))"
        assert len(item) <= 4, f"Box[...] can only be used for x, y, z but got {len(item)} elements"
        dim_order = parse_dim_order(item[0])
        assert len(dim_order) == len(item) - 1, f"Dimension order '{item[0]}' does not match number of slices, {len(item) - 1}"
        lower = []  # None for open sides
        upper = []  # None for open sides
        for dim_name, dim in zip(dim_order, item[1:]):
            assert isinstance(dim, slice)
            assert dim.step is None or dim.step == 1, "Box: step must be 1 but is %s" % dim.step
            lower.append(dim.start)
            upper.append(dim.stop)
        limits = {dim: (lo, up) for dim, lo, up in zip(dim_order, lower, upper)}
        return box_from_limits(**limits)


@sliceable(keepdims='vector')
@dataclass(frozen=True, eq=False)
class Box(Geometry, metaclass=BoxType):
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

    pos: Tensor
    size: Tensor
    rot: Tensor  # can be Layout(None) for no rotation
    is_open: Tensor  # Infinite extent per face, (~side, vector) or fewer
    
    variable_attrs: Tuple[str, ...] = ('pos', 'size', 'rot')

    def __post_init__(self):
        assert isinstance(self.pos, Tensor) and 'vector' in channel(self.pos)
        assert isinstance(self.size, Tensor)
        assert isinstance(self.rot, Tensor)

    @property
    def half_size(self):
        return self.size * 0.5

    @cached_property
    def lower(self):
        return math.where(self.is_open.side['lower'], -math.INF, self.pos - self.half_size)

    @cached_property
    def upper(self):
        return math.where(self.is_open.side['upper'], math.INF, self.pos + self.half_size)

    @cached_property
    def is_finite(self):
        return not self.is_open.any

    def __repr__(self):
        if self.rot is not None:
            return f"Cuboid(center={self.pos}, size={self.size})"
        if self.pos is None or self.size is None:  # traced
            return f"Box[traced, shape={self.shape}]"
        if self.shape.non_channel.volume == 1:
            item_names = self.size.vector.item_names
            if item_names:
                return f"Box({', '.join([f'{dim}=({lo}, {up})' for dim, lo, up in zip(item_names, self.lower, self.upper)])})"
            else:  # deprecated
                return 'Box[%s at %s]' % ('x'.join([str(x) for x in self.size.numpy().flatten()]), ','.join([str(x) for x in self.lower.numpy().flatten()]))
        else:
            return f'Box[shape={self.shape}]'

    @cached_property
    def shape(self):
        return self.pos.shape & self.size.shape & (shape(self.rot) - '~vector')

    @property
    def center(self):
        return self.pos

    @property
    def volume(self) -> Tensor:
        return math.prod(self.size, 'vector')

    @property
    def is_axis_aligned(self):
        return self.rot == None

    @property
    def rotation_matrix(self) -> Tensor:
        return rotation_matrix(self.rot, self.shape['vector'], none_to_unit=True)

    def at(self, center: Tensor) -> 'Box':
        return replace(self, pos=center)

    def rotated(self, angle) -> 'Box':
        rot = wrap(angle) if self.is_axis_aligned.all else self.rotation_matrix @ rotation_matrix(angle)
        return replace(self, rot=rot)

    def scaled(self, factor: Union[float, Tensor]) -> 'Box':
        return replace(self, size=self.size * factor)

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
        pos = rotate(pos, self.rot, invert=True)
        if scale:
            pos /= (self.half_size if origin == 'center' else self.size)
        return pos

    def local_to_global(self, local_position, scale=True, origin='lower'):
        assert origin in ['lower', 'center', 'upper']
        origin_loc = getattr(self, origin)
        pos = local_position * (self.half_size if origin == 'center' else self.size) if scale else local_position
        return rotate(pos, self.rot) + origin_loc

    def __mul__(self, other):
        if not isinstance(other, Box):
            return NotImplemented
        assert self.is_axis_aligned.all and other.is_axis_aligned.all, f"Box * Box only supported for axis-aligned boxes (rot=None)."
        pos = concat([self.pos, other.pos], 'vector')
        size = concat([self.size, other.size], 'vector')
        return replace(self, pos=pos, size=size)

    def bounding_half_extent(self) -> Tensor:
        if self.rot is not None:
            to_face = self.face_normals[{'~side': 0}] * math.rename_dims(self.half_size, 'vector', dual)
            return math.sum(abs(to_face), '~vector')
        return self.half_size

    def lies_inside(self, location: Tensor) -> Tensor:
        union_dims = instance(self) - instance(location)
        location = self.global_to_local(location, scale=False, origin='center')  # scale can only be performed for finite sizes
        if not self.is_open.any:
            bool_inside = abs(location) <= self.half_size
        else:
            above_lower = (location > self.lower) | self.is_open.side.dual['lower']
            below_upper = (location < self.upper) | self.is_open.side.dual['upper']
            bool_inside = above_lower & below_upper
        bool_inside = math.all(bool_inside, 'vector')
        bool_inside = math.any(bool_inside, union_dims)
        return bool_inside

    def largest(self, dim: DimFilter) -> 'Box':
        assert self.is_axis_aligned.all, f"Box.largest() is only supported for axis-aligned boxes (rot=None)"
        dim = self.shape.without('vector').only(dim)
        if not dim:
            return self
        return box_from_limits(math.min(self.lower, dim), math.max(self.upper, dim))

    def smallest(self, dim: DimFilter) -> 'Box':
        assert self.is_axis_aligned.all, f"Box.smallest() is only supported for axis-aligned boxes (rot=None)"
        dim = self.shape.without('vector').only(dim)
        if not dim:
            return self
        return box_from_limits(math.max(self.lower, dim), math.min(self.upper, dim))

    def without(self, dims: Tuple[str, ...]):
        assert self.is_axis_aligned.all, f"Box.without() is only supported for axis-aligned boxes (rot=None)"
        remaining = list(self.shape.get_item_names('vector'))
        for dim in dims:
            if dim in remaining:
                remaining.remove(dim)
        return self.vector[remaining]

    def bounding_radius(self):
        return vec_length(self.half_size)

    def project(self, *dimensions: str):
        """ Project this box into a lower-dimensional space. """
        warnings.warn("Box.project(dims) is deprecated. Use Box.vector[dims] instead", DeprecationWarning, stacklevel=2)
        return self.vector[dimensions]

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
        assert not self.is_open.any, f"approximate_signed_distance not supported for open boxes"
        # ToDo this underestimates diagonally outside points
        location = self.global_to_local(location, scale=False, origin='center')
        distance = math.abs(location) - self.half_size
        distance = math.max(distance, 'vector')
        distance = math.min(distance, self.shape.instance)  # union for instance dimensions
        return distance

    def approximate_closest_surface(self, location: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        assert not self.is_open.any, f"approximate_closest_surface not supported for open boxes"
        loc_to_center = self.global_to_local(location, scale=False, origin='center')
        sgn_surf_delta = math.abs(loc_to_center) - self.half_size
        if instance(self):
            raise NotImplementedError
            # self_center, self_radius, sgn_dist, center_delta, center_dist = math.at_min((self.center, self.radius, sgn_dist, center_delta, center_dist), key=abs(sgn_dist), dim=instance)
        # is_inside = math.all(sgn_surf_delta < 0, 'vector')
        # abs_surf_delta = abs(sgn_surf_delta)
        max_sgn_dist = math.max(sgn_surf_delta, 'vector')
        normal_axis = max_sgn_dist == sgn_surf_delta  # ToDo only one if inside
        normal = math.vec_normalize(normal_axis * math.sign(loc_to_center))
        normal = rotate(normal, self.rot)
        surf_to_center = math.where(normal_axis, math.sign(loc_to_center) * self.half_size, loc_to_center)
        closest_to_center = math.clip(surf_to_center, -self.half_size, self.half_size)
        surface_pos = self.local_to_global(closest_to_center, scale=False, origin='center')
        delta = surface_pos - location
        face_index = expand(0, non_channel(location))
        offset = normal.vector @ surface_pos.vector
        sgn_surf_dist = vec_length(delta) * math.sign(max_sgn_dist)
        return sgn_surf_dist, delta, normal, offset, face_index

    def sample_uniform(self, *shape: Shape) -> Tensor:
        assert not self.is_open.any, f"sample_uniform not supported for open boxes"
        uniform = math.random_uniform(self.shape.non_singleton.without('vector'), *shape, self.shape['vector'])
        return self.lower + uniform * self.size

    def contains(self, other: 'Box'):
        """ Tests if the other box lies fully inside this box. """
        assert not self.is_open.any and not other.is_open.any, f"contains not supported for open boxes"
        assert self.is_axis_aligned.all and other.rot is None, f"contains() is only supported for axis-aligned boxes (rot=None)."
        return np.all(other.lower >= self.lower) and np.all(other.upper <= self.upper)

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
                shift, loc_to_center, rotation_matrix = math.at_min((shift, loc_to_center, rotation_matrix), key=vec_length(shift), dim=instance)
            shift = math.where(abs(shift) > abs(loc_to_center), abs(loc_to_center), shift)  # ensure inward shift ends at center
        shift = rotate(shift, rotation_matrix)
        return positions + math.where(loc_to_center < 0, 1, -1) * shift

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

    @property
    def boundary_elements(self) -> Dict[Any, Dict[str, slice]]:
        return {}

    @property
    def boundary_faces(self) -> Dict[Any, Dict[str, slice]]:
        return {}

    @property
    def faces(self) -> 'Geometry':
        return Cuboid(self.face_centers, self.half_size, self.rot, size_variable=False)

    @property
    def face_centers(self) -> Tensor:
        return self.center + self.face_normals * self.half_size

    @property
    def face_normals(self) -> Tensor:
        unit_vectors = to_float(math.range(self.shape['vector']) == math.range(dual(**self.shape['vector'].untyped_dict)))
        vectors = rotate(unit_vectors, self.rot)
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
    def corners(self) -> Tensor:
        to_face = self.face_normals[{'~side': 'upper'}] * math.rename_dims(self.half_size, 'vector', dual)
        lower_upper = math.meshgrid(math.dual, **{dim: [-1, 1] for dim in self.vector.item_names}, stack_dim=dual('vector'))  # (x=2, y=2, ... vector=x,y,...)
        to_corner = math.sum(lower_upper * to_face, '~vector')
        return self.center + to_corner

    @property
    def is_size_variable(self):
        warnings.warn("Box.is_size_variable is deprecated. Check Box.variable_attrs instead.", DeprecationWarning, stacklevel=2)
        return 'size' in self.variable_attrs

    def corner_representation(self) -> 'Box':
        assert self.is_axis_aligned.all, f"corner_representation does not support rotations"
        return self

    box = corner_representation

    def center_representation(self) -> 'Cuboid':
        return self

    cuboid = center_representation


def box_from_limits(lower: Tensor = None, upper: Tensor = None,
                is_open = wrap(False),
                variable_attrs=('pos', 'size', 'rot'),
                **size: Optional[Union[float, Tensor, tuple, list]]) -> Box:
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
    if upper is not None:
        assert isinstance(upper, Tensor), f"upper must be a Tensor but got {type(upper)}"
        assert 'vector' in upper.shape, "upper must have a vector dimension"
        assert upper.vector.item_names is not None, "vector dimension of lower must list spatial dimension order"
        assert lower is not None, f"upper and lower must be specified together but got None for lower"
        inf_upper = upper
        inf_lower = lower
    else:  # from **size
        lower = []
        upper = []
        for item in size.values():
            if isinstance(item, (tuple, list)):
                assert len(item) == 2, f"Box kwargs must be either dim=upper or dim=(lower,upper) but got {item}"
                lo, up = item
                lower.append(lo)
                upper.append(up)
            elif item is None:
                lower.append(None)
                upper.append(None)
            else:
                lower.append(0)
                upper.append(item)
        vec_dim = channel(vector=list(size))
        any_open = any(is_none(v) for v in lower + upper)
        if any_open:
            inf_lower = stack([-INF if is_none(l) else l for l in lower], vec_dim)
            inf_upper = stack([INF if is_none(u) else u for u in upper], vec_dim)
            lower = where(is_finite(inf_lower), inf_lower, where(is_finite(inf_upper), inf_upper, 0))
            upper = where(is_finite(inf_upper), inf_upper, where(is_finite(inf_lower), inf_lower, 0))
        else:
            upper = inf_upper = stack(upper, vec_dim) if upper else wrap(upper, vec_dim)
            lower = inf_lower = stack(lower, vec_dim) if lower else wrap(lower, vec_dim)
    size = inf_upper - inf_lower
    pos = .5 * (lower + upper)
    # --- Instantiate and set cache ---
    result = Box(pos, size, wrap(None), is_open, variable_attrs)
    if pos.shape in inf_lower.shape:
        result.__dict__['lower'] = inf_lower
    if pos.shape in inf_upper.shape:
        result.__dict__['upper'] = inf_upper
    return result


def Cuboid(center: Tensor = 0,
           half_size: Union[float, Tensor] = None,
           rotation: Optional[Tensor] = None,
           is_open: Tensor = wrap(False),
           variable_attrs=('pos', 'size', 'rot'),
           **size: Union[float, Tensor]) -> Box:
    """
    Args:
        center: Center position
        half_size: Half-size of the cuboid as vector or scalar
        rotation: Rotation angle(s) or rotation matrix.
        is_open: Specify which faces are open, i.e. have infinite extent.
        variable_attrs: Which properties of the box are treated as variable.
        **size: Alternative way of specifying the size. If used, `half_size` must not be specified.
    """
    if half_size is not None:
        assert isinstance(half_size, Tensor), "half_size must be a Tensor"
        assert 'vector' in half_size.shape, f"Cuboid size must have a 'vector' dimension."
        assert half_size.shape.get_item_names('vector') is not None, f"Vector dimension must list spatial dimensions as item names. Use the syntax Cuboid(x=x, y=y) to assign names."
        size = wrap(2 * half_size)
    else:
        size = wrap(tuple(size.values()), math.channel(vector=tuple(size)))
    center = wrap(center)
    if 'vector' not in center.shape or center.shape.get_item_names('vector') is None:
        center = math.expand(center, channel(size))
    rotation = wrap(rotation)
    result = Box(center, size, rotation, is_open, variable_attrs)
    if half_size is not None:
        result.__dict__['half_size'] = half_size
    return result


def bounding_box(geometry: Union[Geometry, Tensor], reduce=non_batch) -> Box:
    """
    Builds a bounding box around `geometry` or a collection of points.

    Args:
        geometry: `Geometry` object or `Tensor` of points.
        reduce: Which objects to includes in each bounding box. Non-reduced dims will be part of the returned box.

    Returns:
        Bounding `Box` containing only batch dims and `vector`.
    """
    if isinstance(geometry, Tensor):
        assert 'vector' in geometry.shape, f"When passing a Tensor to bounding_box, it needs to have a vector dimension but got {geometry.shape}"
        reduce = geometry.shape.only(reduce) - 'vector'
        return Box(math.min(geometry, reduce), math.max(geometry, reduce))
    center = geometry.center
    extent = geometry.bounding_half_extent()
    boxes = Box(center - extent, center + extent)
    return boxes.largest(boxes.shape.only(reduce)-'vector')
