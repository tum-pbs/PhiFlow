from .flag import SAMPLE_POINTS
from .field import Field
from .grid import CenteredGrid
from phi import struct, math


@struct.definition()
class SampledField(Field):

    def __init__(self, name, sample_points, data=1, point_count=None, **kwargs):
        Field.__init__(self, **struct.kwargs(locals(), ignore=['point_count']))
        self._point_count = point_count

    def sample_at(self, points, collapse_dimensions=True):
        raise NotImplementedError()

    def at(self, other_field, collapse_dimensions=True, force_optimization=False, return_self_if_compatible=False):
        if isinstance(other_field, SampledField) and other_field.sample_points is self.sample_points:
            return self
        elif isinstance(other_field, CenteredGrid):
            coords_in_grid = other_field.box.global_to_local(self.data)
            coords_in_grid = coords_in_grid * math.to_float(other_field.resolution) - 0.5
            # TODO not implemented yet
            resampled = math.scatter(coords_in_grid, indices, self.data, shape, duplicates_handling=self.mode)
            return other_field.with_data(resampled * self.data)
        else:
            raise NotImplementedError('SampledField interpolation is only implemented for CenteredGrids.')

    @struct.prop(default='add')
    def mode(self, mode):
        assert mode in ('add', 'mean', 'any')
        return mode

    @struct.attr()
    def sample_points(self, sample_points):
        assert math.ndims(sample_points) == 3
        return sample_points

    @property
    def shape(self):
        with struct.anytype():
            if math.ndims(self.data) > 0:
                data_shape = (self._batch_size, self._point_count, self.component_count)
            else:
                data_shape = ()
            return self.copied_with(data=data_shape,
                                    sample_points=(self._batch_size, self._point_count, self.rank))

    @property
    def rank(self):
        return math.staticshape(self.sample_points)[-1]

    @property
    def component_count(self):
        if math.ndims(self.data) == 0:
            return 1
        return math.shape(self.data)[-1]

    def unstack(self):
        raise NotImplementedError()

    @property
    def points(self):
        if SAMPLE_POINTS in self.flags or self.sample_points is self.data:
            return self
        return SampledField(self.name+'.points', self.sample_points, self.sample_points, flags=[SAMPLE_POINTS])

    def compatible(self, other_field):
        if not other_field.has_points:
            return True
        if isinstance(other_field, SampledField) and other_field.sample_points is self.sample_points:
            return True
        return False

    def __repr__(self):
        return '%s[%sx(%d), %dD]' % (self.__class__.__name__, self._point_count if self._point_count is not None else '?', self.component_count, self.rank)
