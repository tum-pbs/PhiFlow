from __future__ import annotations

from phi import math
from phi.math import Tensor, Shape, spatial_shape, EMPTY_SHAPE


class Geometry:
    """
    Abstract base class for N-dimensional shapes.

    Main implementing classes:

    * Sphere
    * box family: box (generator), Box, Cuboid, AbstractBox

    All geometry objects support batching.
    Thereby any parameter defining the geometry can be varied along arbitrary batch dims.
    All batch dimensions are listed in Geometry.shape.
    """

    @property
    def center(self) -> Tensor:
        """
        :return: center location in single channel dimension, ordered according to GLOBAL_AXIS_ORDER
        :rtype: Tensor
        """
        raise NotImplementedError()

    @property
    def shape(self) -> Shape:
        raise NotImplementedError()

    def unstack(self, dimension):
        raise NotImplementedError()

    @property
    def spatial_rank(self):
        return self.shape.spatial.rank

    def lies_inside(self, location: Tensor) -> Tensor:
        """
        Tests whether the given location lies inside or outside of the geometry. Locations on the surface count as inside.

        :param location: float tensor of shape (batch_size, ..., rank)
        :return: bool tensor of shape (*location.shape[:-1], 1).
        """
        raise NotImplementedError(self.__class__)

    def approximate_signed_distance(self, location: Tensor) -> Tensor:
        """
        Computes the approximate distance from location to the surface of the geometry.
        Locations outside return positive values, inside negative values and zero exactly at the boundary.

        The exact distance metric used depends on the geometry.
        The approximation holds close to the surface and the distance grows to infinity as the location is moved infinitely far from the geometry.
        The distance metric is differentiable and its gradients are bounded at every point in space.

        :param location: float tensor of shape (batch_size, ..., rank)
        :return: float tensor of shape (*location.shape[:-1], 1).
        """
        raise NotImplementedError(self.__class__)

    def approximate_fraction_inside(self, other_geometry: Geometry) -> Tensor:
        """
        Computes the approximate overlap between the geometry and a small other geometry.
        Returns 1.0 if `other_geometry` is fully enclosed in this geometry and 0.0 if there is no overlap.
        Close to the surface of this geometry, the fraction filled is differentiable w.r.t. the location and size of `other_geometry`.

        To call this method on batches of geometries of same shape, pass a batched Geometry instance.
        The result tensor will match the batch shape of `other_geometry`.

        The result may only be accurate in special cases.
        The given geometries may be approximated as spheres or boxes using `bounding_radius()` and `bounding_half_extent()`.

        The default implementation of this method approximates other_geometry as a Sphere and computes the fraction using `approximate_signed_distance()`.

        :param other_geometry: (batched) Geometry instance
        :return: fraction of cell volume lying inside the geometry. float tensor of shape (other_geometry.batch_shape, 1).
        """
        assert isinstance(other_geometry, Geometry)
        radius = other_geometry.bounding_radius()
        location = other_geometry.center
        distance = self.approximate_signed_distance(location)
        inside_fraction = 0.5 - distance / radius
        inside_fraction = math.clip(inside_fraction, 0, 1)
        return inside_fraction

    def bounding_radius(self) -> Tensor:
        """
        Returns the radius of a Sphere object that fully encloses this geometry.
        The sphere is centered at the center of this geometry.

        :return: radius of type float
        """
        raise NotImplementedError(self.__class__)

    def bounding_half_extent(self) -> Tensor:
        """
        The bounding half-extent sets a limit on the outer-most point for each coordinate axis.
        Each component is non-negative.

        Let the bounding half-extent have value `e` in dimension `d` (`extent[...,d] = e`).
        Then, no point of the geometry lies further away from its center point than `e` along `d` (in both axis directions).

        :return: float vector
        """
        raise NotImplementedError(self.__class__)

    def shifted(self, delta: Tensor) -> Geometry:
        """
        Returns a translated version of this geometry.
        :param delta: direction vector
        :return: shifted geometry
        :rtype: Geometry
        """
        raise NotImplementedError(self.__class__)

    def rotated(self, angle) -> Geometry:
        """
        Returns a rotated version of this geometry.
        The geometry is rotated about its center point.

        :param angle: scalar (2d) or vector (3D+) representing delta angle
        :return: rotated geometry
        :rtype: Geometry
        """
        raise NotImplementedError(self.__class__)

    def __invert__(self):
        return _InvertedGeometry(self)

    def __eq__(self, other):
        raise NotImplementedError(self.__class__)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        raise NotImplementedError(self.__class__)


class _InvertedGeometry(Geometry):

    def __init__(self, geometry):
        self.geometry = geometry

    @property
    def center(self):
        raise NotImplementedError()

    @property
    def shape(self):
        return self.geometry.shape

    def lies_inside(self, location: Tensor) -> Tensor:
        return ~self.geometry.lies_inside(location)

    def approximate_signed_distance(self, location: Tensor) -> Tensor:
        return -self.geometry.approximate_signed_distance(location)

    def approximate_fraction_inside(self, other_geometry: Geometry) -> Tensor:
        return 1 - self.geometry.approximate_fraction_inside(other_geometry)

    def bounding_radius(self) -> Tensor:
        raise NotImplementedError()

    def bounding_half_extent(self) -> Tensor:
        raise NotImplementedError()

    def shifted(self, delta: Tensor) -> Geometry:
        return _InvertedGeometry(self.geometry.shifted(delta))

    def rotated(self, angle) -> Geometry:
        return _InvertedGeometry(self.geometry.rotated(angle))

    def unstack(self, dimension):
        return [_InvertedGeometry(g) for g in self.geometry.unstack(dimension)]

    def __eq__(self, other):
        return isinstance(other, _InvertedGeometry) and self.geometry == other.geometry

    def __hash__(self):
        return -hash(self.geometry)


class _NoGeometry(Geometry):

    @property
    def shape(self):
        return EMPTY_SHAPE

    @property
    def center(self):
        return 0

    def bounding_radius(self):
        return 0

    def bounding_half_extent(self):
        return 0

    def approximate_signed_distance(self, location):
        return math.zeros(location.shape.non_channel) + np.inf

    def lies_inside(self, location):
        return math.zeros(location.shape.non_channel, dtype=bool)

    def approximate_fraction_inside(self, other_geometry):
        return math.zeros(other_geometry.shape)

    def shifted(self, delta):
        return self

    def rotated(self, angle):
        return self

    def unstack(self, dimension):
        raise AssertionError('empty geometry cannot be unstacked')

    def __eq__(self, other):
        return isinstance(other, _NoGeometry)

    def __hash__(self):
        return 1


NO_GEOMETRY = _NoGeometry()


def assert_same_rank(rank1, rank2, error_message):
    rank1_, rank2_ = _rank(rank1), _rank(rank2)
    if rank1_ is not None and rank2_ is not None:
        assert rank1_ == rank2_, 'Ranks do not match: %s and %s. %s' % (rank1_, rank2_, error_message)


def _rank(rank):
    if rank is None:
        return None
    elif isinstance(rank, int):
        pass
    elif isinstance(rank, Geometry):
        rank = rank.spatial_rank
    elif isinstance(rank, Shape):
        rank = rank.spatial.rank
    elif isinstance(rank, Tensor):
        rank = rank.shape.spatial_rank
    else:
        raise NotImplementedError(f"{type(rank)} now allowed. Allowed are (int, Geometry, Shape, Tensor).")
    return None if rank == 0 else rank


def _fill_spatial_with_singleton(shape):
    if shape.spatial.rank == shape.channel.volume:
        return shape
    else:
        assert shape.spatial.rank == 0
        return shape.combined(spatial_shape([1] * shape.channel.volume))
