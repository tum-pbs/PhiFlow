from .physics import *
from phi.field import *


GROW = 'grow'
ADD = 'add'
# DECAY = 'decay'
FIX = 'fix'


class FieldEffect(State):

    __struct__ = State.__struct__.extend(('_field', ), ('_mode', '_targets'))

    def __init__(self, field, targets, mode=GROW, tags=('effect',), age=0.0, batch_size=None):
        State.__init__(self, tags=tuple(tags) + tuple('%s_effect' % target for target in targets), age=age, batch_size=batch_size)
        self._field = field
        self._mode = mode
        self._targets = tuple(targets)

    @property
    def field(self):
        return self._field

    @property
    def mode(self):
        return self._mode

    @property
    def targets(self):
        return self._targets

    def __repr__(self):
        return '%s(%s to %s)' % (self.mode, self. field, self.targets)



def effect_applied(effect, field, dt):
    resampled = effect.field.at(field)
    if effect._mode == GROW:
        dt = math.cast(dt, resampled.dtype)
        return field + resampled * dt
    elif effect._mode == ADD:
        return field + resampled
    elif effect._mode == FIX:
        assert effect.field.bounds is not None
        mask = effect.field.bounds.value_at(field.points.data)
        return field * (1 - mask) + resampled * mask
    else:
        raise ValueError('Invalid mode: %s' % effect.mode)


Inflow = lambda geometry, rate=1.0:\
    FieldEffect(GeometryMask('inflow', [geometry], rate), ('density',), GROW, tags=('inflow', 'effect'))
Fan = lambda geometry, acceleration:\
    FieldEffect(GeometryMask('fan', [geometry], acceleration), ('velocity',), GROW, tags=('fan', 'effect'))
ConstantDensity = lambda geometry, density:\
    FieldEffect(GeometryMask('constant-density', [geometry], density), ('density',), FIX)
ConstantTemperature = lambda geometry, temperature:\
    FieldEffect(GeometryMask('constant-temperature', [geometry], temperature), ('temperature',), FIX)
HeatSource = lambda geometry, rate:\
    FieldEffect(GeometryMask('heat-source', [geometry], rate), ('temperature',), GROW)
ColdSource = lambda geometry, rate:\
    FieldEffect(GeometryMask('heat-source', [geometry], -rate), ('temperature',), GROW)


class Gravity(State):

    __struct__ = State.__struct__.extend([], ['_gravity'])

    def __init__(self, gravity=-9.81, batch_size=None):
        State.__init__(self, tags=['gravity'], batch_size=batch_size)
        self._gravity = gravity

    @property
    def gravity(self):
        return self._gravity

    def __add__(self, other):
        if other is 0: return self
        assert isinstance(other, Gravity)
        if self._batch_size is not None: assert self._batch_size == other._batch_size
        # Add gravity
        if math.is_scalar(self._gravity) and math.is_scalar(other._gravity):
            return Gravity(self._gravity + other._gravity)
        else:
            rank = staticshape(other.gravity)[-1] if math.is_scalar(self._gravity) else staticshape(self.gravity)[-1]
            sum_tensor = self.gravity_tensor(rank) + other.gravity_tensor(rank)
            return Gravity(sum_tensor)

    __radd__ = __add__

    def gravity_tensor(self, rank):
        if math.is_scalar(self._gravity):
            return math.expand_dims([self._gravity] + [0] * (rank-1), 0, rank+1)
        else:
            assert staticshape(self._gravity)[-1] == rank
            return math.expand_dims(self._gravity, 0, rank+2-len(staticshape(self._gravity)))