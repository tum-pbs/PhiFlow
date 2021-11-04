from typing import Dict

import numpy as np

from phi import struct, math
from ._geom import Geometry, _fill_spatial_with_singleton
from ._transform import rotate
from ..math import wrap
from ..math._tensors import Tensor
from ..math.backend._backend import combined_dim


class BaseBox(Geometry):  # not a Subwoofer
    """
    Abstract base type for box-like geometries.
    """

    def unstack(self, dimension):
        raise NotImplementedError()

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

    def shifted(self, delta) -> 'BaseBox':
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

    def bounding_radius(self):
        return math.max(self.size, 'vector') * 1.414214

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

    def approximate_signed_distance(self, location):
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
        indices = self.shape.spatial.index(dimensions)
        return Box(self.lower[indices], self.upper[indices])

    def corner_representation(self) -> 'Box':
        return Box(self.lower, self.upper)

    def center_representation(self) -> 'Cuboid':
        return Cuboid(self.center, self.half_size)

    def contains(self, other: 'BaseBox'):
        """ Tests if the other box lies fully inside this box. """
        return np.all(other.lower >= self.lower) and np.all(other.upper <= self.upper)

    def rotated(self, angle):
        return rotate(self, angle)


class BoxType(type):
    """
    Convenience function for creating N-dimensional boxes / cuboids.
    
    Examples to create a box from (0, 0) to (10, 20):
    
    * box[0:10, 0:20]
    * box((0, 0), (10, 20))
    * box((5, 10), size=(10, 20))
    * box(center=(5, 10), size=(10, 20))
    * box((10, 20))

    Args:

    Returns:

    """

    def __getitem__(self, item):
        if not isinstance(item, (tuple, list)):
            item = [item]
        lower = []
        upper = []
        for dim in item:
            assert isinstance(dim, slice)
            assert dim.step is None or dim.step == 1, "Box: step must be 1 but is %s" % dim.step
            lower.append(dim.start if dim.start is not None else -np.inf)
            upper.append(dim.stop if dim.stop is not None else np.inf)
        return Box(lower, upper)


class Box(BaseBox, metaclass=BoxType):
    """
    Simple cuboid defined by location of lower and upper corner in physical space.

    In addition to the regular constructor Box(lower, upper), Box supports construction via slicing, `Box[slice1, slice2,...]`
    Each slice marks the lower and upper edge of the box along one dimension.
    Start and end can be left blank (None) to set the corner point to infinity (upper=None) or -infinity (lower=None).
    The parameter slice.step has no effect.

    **Examples**:

        Box[0:1, 0:1]  # creates a two-dimensional unit box.
        Box[:, 0:1]  # creates an infinite-height Box from x=0 to x=1.
    """

    def __init__(self, lower: Tensor or float or int, upper: Tensor or float or int):
        """
        Args:
          lower: physical location of lower corner
          upper: physical location of upper corner
        """
        self._lower = wrap(lower)
        self._upper = wrap(upper)

    def unstack(self, dimension):
        size = combined_dim(self._lower.shape.get_size(dimension), self._upper.shape.get_size(dimension))
        lowers = self._lower.dimension(dimension).unstack(size)
        uppers = self._upper.dimension(dimension).unstack(size)
        return tuple(Box(lo, up) for lo, up in zip(lowers, uppers))

    def __eq__(self, other):
        return isinstance(other, BaseBox)\
               and set(self.shape) == set(other.shape)\
               and math.close(self._lower, other.lower)\
               and math.close(self._upper, other.upper)

    def __hash__(self):
        return hash(self._upper)

    def __variable_attrs__(self):
        return '_lower', '_upper'

    @property
    def shape(self):
        return _fill_spatial_with_singleton(self._lower.shape & self._upper.shape).non_channel

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @property
    def size(self):
        return self.upper - self.lower

    @struct.derived()
    def center(self):
        return 0.5 * (self.lower + self.upper)

    @struct.derived()
    def half_size(self):
        return self.size * 0.5

    def shifted(self, delta):
        return Box(self.lower + delta, self.upper + delta)

    def __repr__(self):
        if self.shape.non_channel.volume == 1:
            return 'Box[%s at %s]' % ('x'.join([str(x) for x in self.size.numpy().flatten()]), ','.join([str(x) for x in self.lower.numpy().flatten()]))
        else:
            return 'Box[shape=%s]' % self._shape


class Cuboid(BaseBox):

    def __init__(self, center, half_size):
        self._center = wrap(center)
        self._half_size = wrap(half_size)

    def unstack(self, dimension):
        raise NotImplementedError()

    def __eq__(self, other):
        return isinstance(other, BaseBox)\
               and set(self.shape) == set(other.shape)\
               and math.close(self._center, other.center)\
               and math.close(self._half_size, other.half_size)

    def __hash__(self):
        return hash(self._center)

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
        return _fill_spatial_with_singleton(self._center.shape & self._half_size.shape).without('vector')

    @property
    def size(self):
        return 2 * self.half_size

    @property
    def lower(self):
        return self.center - self.half_size

    @property
    def upper(self):
        return self.center + self.half_size

    def shifted(self, delta):
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
        assert resolution.spatial_rank == resolution.rank, 'resolution must be purely spatial but got %s' % (resolution,)
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
    def center(self):
        local_coords = math.meshgrid(**{dim: math.linspace(0.5 / size, 1 - 0.5 / size, size) for dim, size in zip(self.resolution.names, self.resolution.sizes)})
        points = self.bounds.local_to_global(local_coords)
        return points

    @property
    def grid_size(self):
        return self._bounds.size

    @property
    def size(self):
        return self.bounds.size / math.wrap(self.resolution.sizes)

    @property
    def lower(self):
        return self.center - self.half_size

    @property
    def upper(self):
        return self.center + self.half_size

    @property
    def half_size(self):
        return self.bounds.size / self.resolution.sizes / 2

    def __getitem__(self, item: dict):
        bounds = self._bounds
        dx = self.size
        for dim, selection in item.items():
            if dim in self._resolution:
                if isinstance(selection, int):
                    start = selection
                    stop = selection + 1
                elif isinstance(selection, slice):
                    start = selection.start or 0
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
        resolution = self._resolution.after_gather(item)
        return GridCell(resolution, bounds)

    def list_cells(self, dim_name):
        center = math.pack_dims(self.center, self._shape.spatial.names, dim_name)
        return Cuboid(center, self.half_size)

    def stagger(self, dim: str, lower: bool, upper: bool):
        dim_mask = np.array(self.resolution.mask(dim))
        unit = self.bounds.size / self.resolution * dim_mask
        bounds = Box(self.bounds.lower + unit * (-0.5 if lower else 0.5), self.bounds.upper + unit * (0.5 if upper else -0.5))
        ext_res = self.resolution.sizes + dim_mask * (int(lower) + int(upper) - 1)
        return GridCell(self.resolution.with_sizes(ext_res), bounds)

    # def face_centers(self, staggered_name='staggered'):
    #     face_centers = [self.extend_symmetric(dim).center for dim in self.shape.spatial.names]
    #     return math.channel_stack(face_centers, staggered_name)

    @property
    def shape(self):
        return self._shape

    def shifted(self, delta: Tensor) -> BaseBox:
        if delta.shape.spatial_rank == 0:
            return GridCell(self.resolution, self.bounds.shifted(delta))
        else:
            center = self.center + delta
            return Cuboid(center, self.half_size)

    def rotated(self, angle) -> Geometry:
        raise NotImplementedError()

    def unstack(self, dimension):
        raise NotImplementedError()

    def __eq__(self, other):
        return isinstance(other, GridCell) and self._bounds == other._bounds and self._resolution == other._resolution

    def shallow_equals(self, other):
        return self == other

    def __hash__(self):
        return hash(self._resolution) + hash(self._bounds)

    def __repr__(self):
        return f"{self._resolution}, bounds={self._bounds}"
