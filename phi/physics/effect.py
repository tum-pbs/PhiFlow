from .field import *


GROW = 'grow'
ADD = 'add'
DECAY = 'decay'
FIX = 'fix'


class FieldEffect(State):

    __struct__ = State.__struct__.extend(('_field', ), ('_mode', '_targets'))

    def __init__(self, fieldlike, targets, mode=GROW, tags=('effect',), age=0.0, batch_size=None):
        State.__init__(self, tags=tuple(tags) + tuple('%s_effect' % target for target in targets), age=age, batch_size=batch_size)
        self._field = Field.to_field(fieldlike)
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

    def apply_grid(self, target, grid, staggered, dt):
        if self._mode == GROW:
            delta = self.field.sample_grid(grid, staggered=staggered)
            dt = math.cast(dt, math.dtype(delta))
            return target + delta * dt
        elif self._mode == ADD:
            delta = self.field.sample_grid(grid, staggered=staggered)
            return target + delta
        elif self._mode == DECAY:
            raise NotImplementedError()  # TODO
        elif self._mode == FIX:
            assert self.field.bounds is not None
            replacement = self.field.sample_grid(grid, staggered=staggered)
            if staggered:
                raise NotImplementedError()  # TODO
            else:
                mask = 1 - self.field.bounds.at(grid)
                return target * mask + replacement
        else:
            raise ValueError('Invalid mode: %s' % self.mode)

    def __repr__(self):
        return '%s(%s to %s)' % (self.mode, self. field, self.targets)


Inflow = lambda geometry, rate=1.0:\
    FieldEffect(ConstantField(geometry, rate), ('density',), GROW, tags=('inflow', 'effect'))
Fan = lambda geometry, acceleration:\
    FieldEffect(ConstantField(geometry, acceleration), ('velocity',), GROW, tags=('fan', 'effect'))
ConstantDensity = lambda geometry, density:\
    FieldEffect(ConstantField(geometry, density), ('density',), FIX)
ConstantTemperature = lambda geometry, temperature:\
    FieldEffect(ConstantField(geometry, temperature), ('temperature',), FIX)
HeatSource = lambda geometry, rate:\
    FieldEffect(ConstantField(geometry, rate), ('temperature',), GROW)
ColdSource = lambda geometry, rate:\
    FieldEffect(ConstantField(geometry, -rate), ('temperature',), GROW)


class Gravity(State):

    __struct__ = State.__struct__.extend([], ['_gravity'])

    def __init__(self, gravity, batch_size=None):
        State.__init__(self, tags=['gravity'], batch_size=batch_size)
        self._gravity = gravity

    @property
    def gravity(self):
        return self._gravity

    def __add__(self, other):
        assert isinstance(other, Gravity)
        if self._batch_size is not None: assert self._batch_size == other._batch_size
        # Add gravity
        if math.is_scalar(self._gravity) and math.is_scalar(other._gravity):
            return Gravity(self._gravity + other._gravity)
        else:
            rank = staticshape(other.gravity)[-1] if math.is_scalar(self._gravity) else staticshape(self.gravity)[-1]
            sum_tensor = self.gravity_tensor(rank) + other.gravity_tensor(rank)
            return Gravity(sum_tensor)

    def gravity_tensor(self, rank):
        if math.is_scalar(self._gravity):
            return math.expand_dims([self._gravity] + [0] * (rank-1), 0, rank+1)
        else:
            assert staticshape(self._gravity)[-1] == rank
            return math.expand_dims(self._gravity, 0, rank+2-len(staticshape(self._gravity)))