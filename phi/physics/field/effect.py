from phi.physics.field import Field, GeometryMask
from phi import math, struct
from phi.physics import State, Physics, StateDependency

GROW = 'grow'
ADD = 'add'
FIX = 'fix'


@struct.definition()
class FieldEffect(State):

    def __init__(self, field, targets, mode=GROW, tags=('effect',), **kwargs):
        tags = tuple(tags) + tuple('%s_effect' % target for target in targets)
        State.__init__(self, **struct.kwargs(locals()))

    @struct.variable()
    def field(self, field):
        assert isinstance(field, Field)
        return field

    @struct.constant()
    def mode(self, mode):
        assert mode in (GROW, ADD, FIX)
        return mode

    @struct.constant()
    def targets(self, targets):
        return tuple(targets)

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
        raise NotImplementedError()
        # assert effect.field.bounds is not None
        # mask = effect.field.bounds.value_at(field.points.data)
        # return field * (1 - mask) + resampled * mask
    else:
        raise ValueError('Invalid mode: %s' % effect.mode)


Inflow = lambda geometry, rate=1.0: FieldEffect(GeometryMask([geometry], value=rate, name='inflow'), ('density',), GROW, tags=('inflow', 'effect'))
Fan = lambda geometry, acceleration: FieldEffect(GeometryMask([geometry], value=acceleration, name='fan'), ('velocity',), GROW, tags=('fan', 'effect'))
HeatSource = lambda geometry, rate: FieldEffect(GeometryMask([geometry], value=rate, name='heat-source'), ('temperature',), GROW)
ColdSource = lambda geometry, rate: FieldEffect(GeometryMask([geometry], value=-rate, name='heat-source'), ('temperature',), GROW)


@struct.definition()
class Gravity(State):

    def __init__(self, gravity=-9.81, name='gravity', **kwargs):
        tags = ['gravity']
        State.__init__(self, **struct.kwargs(locals()))

    @struct.constant()
    def gravity(self, gravity):
        assert gravity is not None
        return gravity

    def __add__(self, other):
        if other is 0:
            return self
        assert isinstance(other, Gravity), type(other)
        if self._batch_size is not None:
            assert self._batch_size == other._batch_size
        # Add gravity
        if math.is_scalar(self.gravity) and math.is_scalar(other.gravity):
            return Gravity(self.gravity + other.gravity)
        else:
            rank = math.staticshape(other.gravity)[-1] if math.is_scalar(self.gravity)\
                else math.staticshape(self.gravity)[-1]
            sum_tensor = gravity_tensor(self, rank) + gravity_tensor(other, rank)
            return Gravity(sum_tensor)

    __radd__ = __add__


def gravity_tensor(gravity, rank):
    if isinstance(gravity, Gravity):
        gravity = gravity.gravity
    if math.is_scalar(gravity):
        return math.to_float(math.expand_dims([gravity] + [0] * (rank-1), 0, rank+1))
    else:
        assert math.staticshape(gravity)[-1] == rank
        return math.to_float(math.expand_dims(gravity, 0, rank+2-len(math.staticshape(gravity))))


class FieldPhysics(Physics):

    def __init__(self, fieldname):
        Physics.__init__(self, [StateDependency('effects', fieldname+'_effect', blocking=True)])

    def step(self, field, dt=1.0, effects=()):
        for effect in effects:
            field = effect_applied(effect, field, dt)
        return field.copied_with(age = field.age + dt)
