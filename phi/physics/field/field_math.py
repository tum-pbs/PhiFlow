from phi.backend.backend import Backend
from phi import math, struct

from .field import Field
from .analytic import AnalyticField


class SymbolicFieldBackend(Backend):

    def __init__(self):
        Backend.__init__(self, 'Symbolic Field Backend')

    def is_tensor(self, x):
        return isinstance(x, (AnalyticField, _ElementwiseOpField))

    def abs(self, x):
        return _ElementwiseOpField(x, math.abs, ())


@struct.definition()
class _ElementwiseOpField(Field):

    def __init__(self, field, function, function_args):
        Field.__init__(self, None, name='%s(%s)' % (function, field.name))
        self.field = field
        self.function = function
        self.function_args = function_args

    def at(self, other_field, collapse_dimensions=True, force_optimization=False, return_self_if_compatible=False):
        resampled = self.field.at(other_field, collapse_dimensions, force_optimization, return_self_if_compatible)
        applied = self.function(resampled, *self.function_args)
        return applied

    def sample_at(self, points, collapse_dimensions=True):
        pass

    @property
    def rank(self):
        return self.field.rank

    @property
    def component_count(self):
        return self.field.component_count

    def unstack(self):
        return [_ElementwiseOpField(c, self.function_args, self.function_args) for c in self.field.unstack()]

    @property
    def points(self):
        return self.field.points

    def compatible(self, other_field):
        return self.field.compatible(other_field)
