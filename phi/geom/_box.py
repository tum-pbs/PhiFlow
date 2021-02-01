import numpy as np

from phi import struct, math
from ._geom import Geometry, _fill_spatial_with_singleton
from ._transform import rotate
from ..math import tensor
from ..math._tensors import Tensor


class AbstractBox(Geometry):

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
        """
        Center point of the geometry or geometry batch.
        
        The shape of the location extends the shape of the Geometry by a `vector` dimension.
        
        :return: Tensor describing the center location(s)

        Args:

        Returns:

        """
        raise NotImplementedError()

    def shifted(self, delta) -> 'AbstractBox':
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
        return math.all(bool_inside, 'vector')

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
        return math.max(distance, 'vector')

    def push(self, positions: Tensor, outward: bool = True, shift_amount: float = 0) -> Tensor:
        """
        Shifts positions either into or out of geometry.

        Args:
            positions: Tensor holding positions to shift
            outward: Flag for indicating inward (False) or outward (True) shift
            shift_amount: Offset to box boundaries after shifting

        Returns:
            Tensor holding shifted positions
        """
        center = 0.5 * (self.lower + self.upper)
        extent = self.upper - self.lower
        loc_to_center = positions - center
        distance = math.abs(loc_to_center) - extent * 0.5
        if outward:
            shift = math.where(distance == math.max(distance, 'vector'), distance, 0)  # get shift towards nearest border
            shift = math.where(shift < 0, shift, 0)  # filter points inside geometry
            shift = math.where(loc_to_center < 0, 1, -1) * (shift - math.where(shift != 0, shift_amount, 0))  # get shift direction
        else:
            shift = math.where(distance < 0, 0, distance)
            shift += math.where(shift != 0, shift_amount, 0)
            shift = math.where(math.abs(shift) > math.abs(loc_to_center), math.abs(loc_to_center), shift)  # ensure inward shift ends at center
            shift = math.where(loc_to_center < 0, 1, -1) * shift  # get shift direction
        return positions + shift

    def project(self, *dimensions: str):
        """ Project this box into a lower-dimensional space. """
        indices = self.shape.spatial.index(dimensions)
        return Box(self.lower[indices], self.upper[indices])

    def corner_representation(self) -> 'Box':
        return Box(self.lower, self.upper)

    def center_representation(self) -> 'Cuboid':
        return Cuboid(self.center, self.half_size)

    def contains(self, other: 'AbstractBox'):
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


class Box(AbstractBox, metaclass=BoxType):
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
        self._lower = tensor(lower)
        self._upper = tensor(upper)
        self._shape = _fill_spatial_with_singleton(self._lower.shape & self._upper.shape)

    def unstack(self, dimension):
        raise NotImplementedError()  # TODO

    def __eq__(self, other):
        return isinstance(other, AbstractBox) and self._lower.shape == other.lower.shape and math.close(self._lower, other.lower)

    def __hash__(self):
        return hash(self._upper)

    @property
    def shape(self):
        return self._shape

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

    def without_axis(self, axis):
        lower = []
        upper = []
        for ax in range(self.spatial_rank):
            if ax != axis:
                lower.append(self.get_lower(ax))
                upper.append(self.get_upper(ax))
        return Box(lower, upper)

    def shifted(self, delta):
        return Box(self.lower + delta, self.upper + delta)

    def __repr__(self):
        if self.shape.non_channel.volume == 1:
            return 'Box[%s at %s]' % ('x'.join([str(x) for x in self.size.numpy().flatten()]), ','.join([str(x) for x in self.lower.numpy().flatten()]))
        else:
            return 'Box[shape=%s]' % self._shape


class Cuboid(AbstractBox):

    def __init__(self, center, half_size):
        self._center = tensor(center, names='..., vector', channel_dims=1, spatial_dims=0)
        self._half_size = tensor(half_size, names='..., vector', channel_dims=1, spatial_dims=0)
        self._shape = _fill_spatial_with_singleton(self._center.shape & self._half_size.shape).without('vector')

    @property
    def center(self):
        return self._center

    @property
    def half_size(self):
        return self._half_size

    @property
    def shape(self):
        return self._shape

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


class GridCell(AbstractBox):

    def __init__(self, resolution: math.Shape, bounds: AbstractBox):
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
        """
        Center point of the box or batch of boxes.
        
        The shape of the location extends the shape of the Box instance by a `vector` dimension.
        
        :return: Tensor describing the center location(s)

        Args:

        Returns:

        """
        local_coords = math.meshgrid(**{dim: math.linspace(0.5 / size, 1 - 0.5 / size, size) for dim, size in self.resolution.named_sizes})
        points = self.bounds.local_to_global(local_coords)
        return points

    @property
    def grid_size(self):
        return self._bounds.size

    @property
    def size(self):
        return self.bounds.size / self.resolution.sizes

    @property
    def lower(self):
        return self.center - self.half_size

    @property
    def upper(self):
        return self.center + self.half_size

    @property
    def half_size(self):
        return self.bounds.size / self.resolution.sizes / 2

    def list_cells(self, dim_name):
        center = math.join_dimensions(self.center, self._shape.spatial.names, dim_name)
        return Cuboid(center, self.half_size)

    def extend_symmetric(self, dims: str or list or tuple, cells: int):
        axis_mask = np.array(self.resolution.mask(dims)) * cells
        unit = self.bounds.size / self.resolution * axis_mask
        delta_size = unit / 2
        bounds = Box(self.bounds.lower - delta_size, self.bounds.upper + delta_size)
        ext_res = self.resolution.sizes + axis_mask
        return GridCell(self.resolution.with_sizes(ext_res), bounds)

    def face_centers(self, staggered_name='staggered'):
        face_centers = [self.extend_symmetric(dim, 1).center for dim in self.shape.spatial.names]
        return math.channel_stack(face_centers, staggered_name)

    @property
    def shape(self):
        return self._shape

    def shifted(self, delta: Tensor) -> 'GridCell':
        return GridCell(self.resolution, self.bounds.shifted(delta))

    def rotated(self, angle) -> Geometry:
        raise NotImplementedError()

    def unstack(self, dimension):
        raise NotImplementedError()

    def __eq__(self, other):
        return isinstance(other, GridCell) and self._bounds == other._bounds and self._resolution == other._resolution

    def __hash__(self):
        return hash(self._resolution) + hash(self._bounds)

    def __repr__(self):
        return f"{self._resolution}, bounds={self._bounds}"
