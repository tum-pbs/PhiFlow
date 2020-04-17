import warnings

from phi import struct, math, geom

from .field import Field
from .analytic import AnalyticField


@struct.definition()
class GeometryMask(AnalyticField):

    def __init__(self, geometries, antialias=False, value=None, **kwargs):
        if value is not None:
            warnings.warn('Passing a value to GeometryMask has no effect.')
        rank = geometries.rank if isinstance(geometries, geom.Geometry) else geometries[0].rank
        AnalyticField.__init__(self, rank, **struct.kwargs(locals(), ignore=['value', 'rank']))

    @struct.constant()
    def geometries(self, geometry):
        """ Alias for `geometry`. """
        if isinstance(geometry, (tuple, list)):
            geometry = geom.union(geometry)
        assert isinstance(geometry, geom.Geometry)
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

    def approximate_mean_value_in(self, geometry):
        if not self.antialias:
            return Field.approximate_mean_value_in(self, geometry)
        else:
            return self.geometry.approximate_fraction_inside(geometry)

    def sample_at(self, points):
        return math.to_float(self.geometry.lies_inside(points))

    @property
    def component_count(self):
        return 1


mask = GeometryMask


def union_mask(geometries):
    warnings.warn("union_mask() is deprecated, use mask(union()) instead.", DeprecationWarning)
    return mask(geom.union(*geometries))
