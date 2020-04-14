import warnings

from phi import struct, math
from phi.geom import Geometry, union

from .field import Field
from .grid import CenteredGrid
from .staggered_grid import StaggeredGrid
from .analytic import AnalyticField


@struct.definition()
class GeometryMask(AnalyticField):

    def __init__(self, geometries, value=None, **kwargs):
        if value is not None:
            warnings.warn('Passing a value to GeometryMask has no effect.')
        rank = geometries.rank if isinstance(geometries, Geometry) else geometries[0].rank
        AnalyticField.__init__(self, rank, **struct.kwargs(locals(), ignore=['value', 'rank']))

    @struct.constant()
    def geometries(self, geometry):
        """ Alias for `geometry`. """
        if isinstance(geometry, (tuple, list)):
            geometry = union(geometry)
        assert isinstance(geometry, Geometry)
        return geometry

    geometry = geometries

    @struct.constant(default=False)
    def antialias(self, antialias):
        """
If False, field values are either 0 (outside) or 1 (inside) and the field is not differentiable w.r.t. the geometry.
If True, field values smoothly go from 0 to 1 at the surface and the field is differentiable w.r.t. the geometry.
        """
        assert antialias in (True, False)
        return antialias

    @struct.constant(default=1)
    def default_cell_size(self, size):
        return math.to_float(size)

    def sample_at(self, points):
        if not self.antialias:
            return math.to_float(self.geometry.lies_inside(points))
        else:
            return self.geometry.approximate_fraction_inside(points, self.default_cell_size)

    def at(self, other_field):
        geometry_mask = self
        if isinstance(other_field, (CenteredGrid, StaggeredGrid)):
            geometry_mask = self.copied_with(default_cell_size=math.mean(other_field.dx))
        return Field.at(geometry_mask, other_field)

    @property
    def component_count(self):
        return 1


mask = GeometryMask


def union_mask(geometries):
    warnings.warn("union_mask() is deprecated, use mask(union()) instead.", DeprecationWarning)
    return mask(union(*geometries))
