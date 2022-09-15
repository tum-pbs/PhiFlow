import warnings
from typing import Dict, Tuple

import numpy as np

from phi import math
from ._geom import Geometry, _keep_vector
from ..math import wrap, INF, Shape, channel, spatial, copy_with, Tensor
from ..math._shape import parse_dim_order
from ..math.magic import slicing_dict


class BaseBox(Geometry):  # not a Subwoofer
    """
    Abstract base type for box-like geometries.
    """

    def __eq__(self, other):
        raise NotImplementedError()

    def __hash__(self):
        raise NotImplementedError()

    def __ne__(self, other):
        return not self == other

    @property
    def shape(self):
        raise NotImplementedError()

    @property
    def center(self) -> Tensor:
        raise NotImplementedError()

    def shifted(self, delta, **delta_by_dim) -> 'BaseBox':
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
    def volume(self) -> Tensor:
        return math.prod(self.size, 'vector')

    @property
    def shape_type(self) -> Tensor:
        return math.tensor('B')

    def bounding_radius(self):
        return math.vec_length(self.half_size)

    def bounding_half_extent(self):
        return self.size * 0.5

    def global_to_local(self, global_position: Tensor) -> Tensor:
        if math.close(self.lower, 0):
            return global_position / self.size
        else:
            return (global_position - self.lower) / self.size

    def local_to_global(self, local_position):
        return local_position * self.size + self.lower

    def lies_inside(self, location):
        bool_inside = (location >= self.lower) & (location <= self.upper)
        bool_inside = math.all(bool_inside, 'vector')
        bool_inside = math.any(bool_inside, self.shape.instance)  # union for instance dimensions
        return bool_inside

    def approximate_signed_distance(self, location: Tensor or tuple):
        """
        Computes the signed L-infinity norm (manhattan distance) from the location to the nearest side of the box.
        For an outside location `l` with the closest surface point `s`, the distance is `max(abs(l - s))`.
        For inside locations it is `-max(abs(l - s))`.

        Args:
          location: float tensor of shape (batch_size, ..., rank)

        Returns:
          float tensor of shape (*location.shape[:-1], 1).

        """
        center = 0.5 * (self.lower + self.upper)
        extent = self.upper - self.lower
        distance = math.abs(location - center) - extent * 0.5
        distance = math.max(distance, 'vector')
        distance = math.min(distance, self.shape.instance)  # union for instance dimensions
        return distance

    def push(self, positions: Tensor, outward: bool = True, shift_amount: float = 0) -> Tensor:
        loc_to_center = positions - self.center
        sgn_dist_from_surface = math.abs(loc_to_center) - self.half_size
        if outward:
            # --- get negative distances (particles are inside) towards the nearest boundary and add shift_amount ---
            distances_of_interest = (sgn_dist_from_surface == math.max(sgn_dist_from_surface, 'vector')) & (sgn_dist_from_surface < 0)
            shift = distances_of_interest * (sgn_dist_from_surface - shift_amount)
        else:
            shift = (sgn_dist_from_surface + shift_amount) * (sgn_dist_from_surface > 0)  # get positive distances (particles are outside) and add shift_amount
            shift = math.where(math.abs(shift) > math.abs(loc_to_center), math.abs(loc_to_center), shift)  # ensure inward shift ends at center
        return positions + math.where(loc_to_center < 0, 1, -1) * shift

    def project(self, *dimensions: str):
        """ Project this box into a lower-dimensional space. """
        warnings.warn("Box.project(dims) is deprecated. Use Box.vector[dims] instead", DeprecationWarning, stacklevel=2)
        return self.vector[dimensions]

    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        uniform = math.random_uniform(self.shape.non_singleton, *shape, math.channel(vector=self.spatial_rank))
        return self.lower + uniform * self.size

    def corner_representation(self) -> 'Box':
        return Box(self.lower, self.upper)

    box = corner_representation

    def center_representation(self) -> 'Cuboid':
        return Cuboid(self.center, self.half_size)

    def contains(self, other: 'BaseBox'):
        """ Tests if the other box lies fully inside this box. """
        return np.all(other.lower >= self.lower) and np.all(other.upper <= self.upper)

    def rotated(self, angle) -> Geometry:
        from ._transform import rotate
        return rotate(self, angle)

    def scaled(self, factor: float or Tensor) -> 'Geometry':
        return Cuboid(self.center, self.half_size * factor)


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

    **Examples**:

    ```python
    Box(x=1, y=1)  # creates a two-dimensional unit box with `lower=(0, 0)` and `upper=(1, 1)`.
    Box(x=(None, 1), y=(0, None)  # creates a Box with `lower=(-inf, 0)` and `upper=(1, inf)`.
    ```

    The slicing constructor was updated in version 2.2 and now requires the dimension order as the first argument.

    ```python
    Box['x,y', 0:1, 0:1]  # creates a two-dimensional unit box with `lower=(0, 0)` and `upper=(1, 1)`.
    Box['x,y', :1, 0:]  # creates a Box with `lower=(-inf, 0)` and `upper=(1, inf)`.
    ```
    """

    def __init__(self, lower: Tensor = None, upper: Tensor = None, **size: int or Tensor):
        """
        Args:
          lower: physical location of lower corner
          upper: physical location of upper corner
          **size: Specify size by dimension, either as `int` or `tuple` containing (lower, upper).
        """
        if lower is not None:
            assert isinstance(lower, Tensor)
            assert 'vector' in lower.shape, "lower must have a vector dimension"
            assert lower.vector.item_names is not None, "vector dimension of lower must list spatial dimension order"
            self._lower = lower
        if upper is not None:
            assert isinstance(upper, Tensor)
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

    def __getitem__(self, item):
        item = _keep_vector(slicing_dict(self, item))
        return Box(self._lower[item], self._upper[item])

    def __stack__(self, values: tuple, dim: Shape, **kwargs) -> 'Geometry':
        if all(isinstance(v, Box) for v in values):
            return Box(math.stack([v.lower for v in values], dim, **kwargs), math.stack([v.upper for v in values], dim, **kwargs))
        else:
            return Geometry.__stack__(self, values, dim, **kwargs)

    def __eq__(self, other):
        return isinstance(other, BaseBox)\
               and set(self.shape) == set(other.shape)\
               and self.size.shape.get_size('vector') == other.size.shape.get_size('vector')\
               and math.close(self._lower, other.lower)\
               and math.close(self._upper, other.upper)

    def without(self, dims: Tuple[str, ...]):
        remaining = list(self.shape.get_item_names('vector'))
        for dim in dims:
            if dim in remaining:
                remaining.remove(dim)
        return self.vector[remaining]

    def __hash__(self):
        return hash(self._upper)

    def __variable_attrs__(self):
        return '_lower', '_upper'

    @property
    def shape(self):
        if self._lower is None or self._upper is None:
            return None
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

    def shifted(self, delta, **delta_by_dim):
        return Box(self.lower + delta, self.upper + delta)

    def __mul__(self, other):
        if not isinstance(other, Box):
            return NotImplemented
        lower = self._lower.vector.unstack(self.spatial_rank) + other._lower.vector.unstack(other.spatial_rank)
        upper = self._upper.vector.unstack(self.spatial_rank) + other._upper.vector.unstack(other.spatial_rank)
        names = self._upper.vector.item_names + other._upper.vector.item_names
        lower = math.stack(lower, math.channel(vector=names))
        upper = math.stack(upper, math.channel(vector=names))
        return Box(lower, upper)

    def __repr__(self):
        if self.shape.non_channel.volume == 1:
            item_names = self.size.vector.item_names
            if item_names:
                return f"Box({', '.join([f'{dim}=({lo}, {up})' for dim, lo, up in zip(item_names, self._lower, self._upper)])})"
            else:  # deprecated
                return 'Box[%s at %s]' % ('x'.join([str(x) for x in self.size.numpy().flatten()]), ','.join([str(x) for x in self.lower.numpy().flatten()]))
        else:
            return f'Box[shape={self.shape}]'


class Cuboid(BaseBox):
    """
    Box specified by center position and half size.
    """

    def __init__(self,
                 center: Tensor = 0,
                 half_size: float or Tensor = None,
                 **size: float or Tensor):
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


    def __eq__(self, other):
        return isinstance(other, BaseBox)\
               and set(self.shape) == set(other.shape)\
               and math.close(self._center, other.center)\
               and math.close(self._half_size, other.half_size)

    def __hash__(self):
        return hash(self._center)

    def __getitem__(self, item):
        item = _keep_vector(slicing_dict(self, item))
        return Cuboid(self._center[item], self._half_size[item])

    def __stack__(self, values: tuple, dim: Shape, **kwargs) -> 'Geometry':
        if all(isinstance(v, Cuboid) for v in values):
            return Cuboid(math.stack([v.center for v in values], dim, **kwargs), math.stack([v.half_size for v in values], dim, **kwargs))
        else:
            return Geometry.__stack__(self, values, dim, **kwargs)

    def __variable_attrs__(self):
        return '_center', '_half_size'

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
        return 2 * self.half_size

    @property
    def lower(self):
        return self.center - self.half_size

    @property
    def upper(self):
        return self.center + self.half_size

    def shifted(self, delta, **delta_by_dim) -> 'Cuboid':
        return Cuboid(self._center + delta, self._half_size)


def bounding_box(geometry):
    center = geometry.center
    extent = geometry.bounding_half_extent()
    return Box(lower=center - extent, upper=center + extent)


class GridCell(BaseBox):
    """
    An instance of GridCell represents all cells of a regular grid as a batch of boxes.
    """

    def __init__(self, resolution: math.Shape, bounds: BaseBox):
        assert resolution.spatial_rank == resolution.rank, f"resolution must be purely spatial but got {resolution}"
        assert resolution.spatial_rank == bounds.spatial_rank, f"bounds must match dimensions of resolution but got {bounds} for resolution {resolution}"
        self._resolution = resolution
        self._bounds = bounds
        self._shape = resolution & bounds.shape.non_spatial

    @property
    def resolution(self):
        return self._resolution

    @property
    def bounds(self):
        return self._bounds

    @property
    def spatial_rank(self) -> int:
        return self._resolution.spatial_rank

    @property
    def center(self):
        local_coords = math.meshgrid(**{dim.name: math.linspace(0.5 / dim.size, 1 - 0.5 / dim.size, dim) for dim in self.resolution})
        points = self.bounds.local_to_global(local_coords)
        return points

    @property
    def grid_size(self):
        return self._bounds.size

    @property
    def size(self):
        return self.bounds.size / math.wrap(self.resolution.sizes)

    @property
    def dx(self):
        return self.bounds.size / self.resolution

    @property
    def lower(self):
        return self.center - self.half_size

    @property
    def upper(self):
        return self.center + self.half_size

    @property
    def half_size(self):
        return self.bounds.size / self.resolution.sizes / 2

    def __getitem__(self, item):
        item = slicing_dict(self, item)
        bounds = self._bounds
        dx = self.size
        gather_dict = {}
        for dim, selection in item.items():
            if dim in self._resolution:
                if isinstance(selection, int):
                    start = selection
                    stop = selection + 1
                elif isinstance(selection, slice):
                    start = selection.start or 0
                    if start < 0:
                        start += self.resolution.get_size(dim)
                    stop = selection.stop or self.resolution.get_size(dim)
                    if stop < 0:
                        stop += self.resolution.get_size(dim)
                    assert selection.step is None or selection.step == 1
                else:
                    raise ValueError(f"Illegal selection: {item}")
                dim_mask = math.wrap(self.resolution.mask(dim))
                lower = bounds.lower + start * dim_mask * dx
                upper = bounds.upper + (stop - self.resolution.get_size(dim)) * dim_mask * dx
                bounds = Box(lower, upper)
                gather_dict[dim] = slice(start, stop)
        resolution = self._resolution.after_gather(gather_dict)
        return GridCell(resolution, bounds)

    def __pack_dims__(self, dims: Tuple[str, ...], packed_dim: Shape, pos: int or None, **kwargs) -> 'Cuboid':
        return math.pack_dims(self.center_representation(), dims, packed_dim, pos, **kwargs)

    def list_cells(self, dim_name):
        center = math.pack_dims(self.center, self._shape.spatial.names, dim_name)
        return Cuboid(center, self.half_size)

    def stagger(self, dim: str, lower: bool, upper: bool):
        dim_mask = np.array(self.resolution.mask(dim))
        unit = self.bounds.size / self.resolution * dim_mask
        bounds = Box(self.bounds.lower + unit * (-0.5 if lower else 0.5), self.bounds.upper + unit * (0.5 if upper else -0.5))
        ext_res = self.resolution.sizes + dim_mask * (int(lower) + int(upper) - 1)
        return GridCell(self.resolution.with_sizes(ext_res), bounds)

    def padded(self, widths: dict):
        resolution, bounds = self.resolution, self.bounds
        for dim, (lower, upper) in widths.items():
            masked_dx = self.dx * math.dim_mask(self.resolution, dim)
            resolution = resolution.with_dim_size(dim, self.resolution.get_size(dim) + lower + upper)
            bounds = Box(bounds.lower - masked_dx * lower, bounds.upper + masked_dx * upper)
        return GridCell(resolution, bounds)

    # def face_centers(self, staggered_name='staggered'):
    #     face_centers = [self.extend_symmetric(dim).center for dim in self.shape.spatial.names]
    #     return math.channel_stack(face_centers, staggered_name)

    @property
    def shape(self):
        return self._shape

    def shifted(self, delta: Tensor, **delta_by_dim) -> BaseBox:
        # delta += math.padded_stack()
        if delta.shape.spatial_rank == 0:
            return GridCell(self.resolution, self.bounds.shifted(delta))
        else:
            center = self.center + delta
            return Cuboid(center, self.half_size)

    def rotated(self, angle) -> Geometry:
        raise NotImplementedError("Grids cannot be rotated. Use center_representation() to convert it to Cuboids first.")

    def __eq__(self, other):
        return isinstance(other, GridCell) and self._bounds == other._bounds and self._resolution == other._resolution

    def shallow_equals(self, other):
        return self == other

    def __hash__(self):
        return hash(self._resolution) + hash(self._bounds)

    def __repr__(self):
        return f"{self._resolution}, bounds={self._bounds}"

    def __variable_attrs__(self):
        return '_center', '_half_size'

    def __with_attrs__(self, **attrs):
        return copy_with(self.center_representation(), **attrs)

    @property
    def _center(self):
        return self.center

    @property
    def _half_size(self):
        return self.half_size
