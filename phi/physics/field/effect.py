import warnings

import six

from phi.geom import Geometry, GLOBAL_AXIS_ORDER
from phi.physics.field import Field, mask, ConstantField
from phi import math, struct
from phi.physics import State, Physics, StateDependency

GROW = 'grow'
ADD = 'add'
FIX = 'fix'


@struct.definition()
class FieldEffect(State):

    def __init__(self, field, targets, mode=GROW, bounds=None, tags=('effect',), **kwargs):
        if isinstance(targets, six.string_types):
            targets = [targets]
        tags = tuple(tags) + tuple('%s_effect' % target for target in targets)
        State.__init__(self, **struct.kwargs(locals()))

    @struct.variable()
    def field(self, field):
        assert isinstance(field, Field) or field is None, field
        return field

    @struct.constant()
    def mode(self, mode):
        assert mode in (GROW, ADD, FIX)
        return mode

    @struct.constant()
    def targets(self, targets):
        return tuple(targets)

    @struct.constant()
    def bounds(self, bounds):
        assert isinstance(bounds, Geometry) or bounds is None
        return bounds

    def __repr__(self):
        return '%s(%s to %s)' % (self.mode, self. field, self.targets)


def effect_applied(effect, field, dt):
    effect_field = effect.field.at(field)
    if effect._mode == GROW:
        return field + math.mul(effect_field, dt)
    elif effect._mode == ADD:
        return field + effect_field
    elif effect._mode == FIX:
        assert effect.bounds is not None
        inside = mask([effect.bounds]).at(field)
        return math.where(inside, effect_field, field)
    else:
        raise ValueError('Invalid mode: %s' % effect.mode)


# pylint: disable-msg = invalid-name
def Inflow(geometry, rate=1.0, target='density'): return FieldEffect(mask(geometry, antialias=True) * rate, target, GROW, tags=('inflow', 'effect'))
def Accelerator(geometry, acceleration): return FieldEffect(mask(geometry, antialias=True) * acceleration, ('velocity',), GROW, tags=('fan', 'effect'))
def ConstantVelocity(geometry, velocity): return FieldEffect(ConstantField(velocity), bounds=geometry, targets=('velocity',), mode=FIX, tags=('effect',))
def HeatSource(geometry, rate, name=None): return FieldEffect(mask(geometry, antialias=True) * rate, ('temperature',), GROW, name=name)
def ColdSource(geometry, rate, name=None): return FieldEffect(mask(geometry, antialias=True) * -rate, ('temperature',), GROW, name=name)


def Fan(*args, **kwargs):
    # pylint: disable-msg = invalid-name
    warnings.warn("'Fan' was renamed to 'Accelerator' in version 1.0.2.", DeprecationWarning)
    return Accelerator(*args, **kwargs)


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
        gravity = gravity * GLOBAL_AXIS_ORDER.up_vector(rank)
    assert math.staticshape(gravity)[-1] == rank
    return math.to_float(math.expand_dims(gravity, 0, rank + 2 - len(math.staticshape(gravity))))


class FieldPhysics(Physics):

    def __init__(self, fieldname):
        Physics.__init__(self, [StateDependency('effects', fieldname + '_effect', blocking=True)])

    def step(self, field, dt=1.0, effects=()):
        for effect in effects:
            field = effect_applied(effect, field, dt)
        return field.copied_with(age=field.age + dt)
