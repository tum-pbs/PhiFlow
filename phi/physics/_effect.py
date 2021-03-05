import warnings

from phi.geom import Geometry
from phi.field import Field, GeometryMask, ConstantField, SampledField
from phi import math, struct
from ._physics import State, Physics, StateDependency

GROW = 'grow'
ADD = 'add'
FIX = 'fix'


@struct.definition()
class FieldEffect(State):

    def __init__(self, field, targets, factor=1., mode=GROW, bounds=None, tags=('effect',), **kwargs):
        if isinstance(targets, str):
            targets = [targets]
        tags = tuple(tags) + tuple('%s_effect' % target for target in targets)
        State.__init__(self, **struct.kwargs(locals()))

    @struct.variable()
    def field(self, field):
        assert isinstance(field, Field) or field is None, field
        return field

    @struct.constant()
    def factor(self, factor):
        return factor

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


def effect_applied(effect: FieldEffect, field: SampledField, dt):
    effect_field = effect.field.at(field) * effect.factor
    if effect._mode == GROW:
        return field + effect_field * dt
    elif effect._mode == ADD:
        return field + effect_field
    elif effect._mode == FIX:
        assert effect.bounds is not None
        inside = GeometryMask(effect.bounds).at(field)
        return inside * effect_field + (1 - inside) * field
    else:
        raise ValueError('Invalid mode: %s' % effect.mode)


# pylint: disable-msg = invalid-name
Inflow = lambda geometry, rate=1.0, target='density': FieldEffect(GeometryMask(geometry), target, rate, GROW, tags=('inflow', 'effect'))
Accelerator = lambda geometry, acceleration: FieldEffect(GeometryMask(geometry), ('velocity',), acceleration, GROW, tags=('fan', 'effect'))
ConstantVelocity = lambda geometry, velocity: FieldEffect(ConstantField(velocity), bounds=geometry, targets=('velocity',), mode=FIX, tags=('effect',))
HeatSource = lambda geometry, rate, name=None: FieldEffect(GeometryMask(geometry), ('temperature',), rate, GROW, name=name)
ColdSource = lambda geometry, rate, name=None: FieldEffect(GeometryMask(geometry), ('temperature',), -rate, GROW, name=name)


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


def gravity_tensor(gravity, rank):
    if isinstance(gravity, Gravity):
        gravity = gravity.gravity
    return gravity * math.GLOBAL_AXIS_ORDER.up_vector(rank)


class FieldPhysics(Physics):

    def __init__(self, fieldname):
        Physics.__init__(self, [StateDependency('effects', fieldname+'_effect', blocking=True)])

    def step(self, field, dt=1.0, effects=()):
        for effect in effects:
            field = effect_applied(effect, field, dt)
        return field.copied_with(age=field.age + dt)
