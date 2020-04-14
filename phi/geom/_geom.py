import warnings

from phi import struct
from phi import math


@struct.definition()
class Geometry(struct.Struct):

    def value_at(self, location):
        """
Samples the geometry at the given locations and returns a binary mask, labelling the points as inside=1, outside=0.
        :param location: tensor of shape (batch_size, ..., rank)
        :return: float tensor of shape (*location.shape[:-1], 1).
        """
        warnings.warn("Geometry.value_at() is deprecated. Use lies_inside or approximate_fraction_inside instead.", DeprecationWarning)
        return math.to_float(self.lies_inside(location))

    def lies_inside(self, location):
        """
Tests whether the given location lies inside or outside of the geometry. Locations on the surface count as inside.
        :param location: float tensor of shape (batch_size, ..., rank)
        :return: bool tensor of shape (*location.shape[:-1], 1).
        """
        raise NotImplementedError(self.__class__)

    def approximate_signed_distance(self, location):
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

    def approximate_fraction_inside(self, location, cell_size):
        """
Computes the approximate overlap between the geometry and small cells.
Cells that lie completely inside the geometry return 1.0, those that lie completely outside return 0.0.
Close to the geometry surface, the fraction filled is differentiable w.r.t. the cell location and size.

No specific cell shape is assumed. Cells may be approximated as spheres or axis-aligned cubes.

Cell sizes should rather be overestimated than underestimated to avoid zero gradients.
        :param location: float tensor of shape (batch_size, ..., rank)
        :param cell_size: length or diameter of each cell. Scalar or tensor of shape compatible with location.
        :return: fraction of cell volume lying inside the geometry. float tensor of shape (*location.shape[:-1], 1).
        """
        radius = 0.707 * cell_size
        distance = self.approximate_signed_distance(location)
        inside_fraction = 0.5 - distance / radius
        inside_fraction = math.clip(inside_fraction, 0, 1)
        return inside_fraction

    @property
    def rank(self):
        raise NotImplementedError(self.__class__)
