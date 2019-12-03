from .field import Field
from phi import struct, math


@struct.definition()
class AnalyticField(Field):

    def __init__(self, rank, data=1.0, name=None, **kwargs):
        Field.__init__(self, **struct.kwargs(locals(), ignore='rank'))
        self._rank = rank

    @property
    def rank(self):
        return self._rank

    @property
    def component_count(self):
        example_value = self.sample_at([[0] * self._rank])
        return math.staticshape(example_value)[-1]

    def unstack(self):
        if self.component_count == 1:
            return [self]
        else:
            raise NotImplementedError()

    @property
    def points(self):
        return None

    def compatible(self, other_field):
        return True
