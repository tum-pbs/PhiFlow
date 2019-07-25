from .field import *


GROW = 'add'
DECAY = 'mul'
FIX = 'replace'


class FieldEffect(State):

    __struct__ = State.__struct__.extend((), ('_field', '_mode', '_targets'))

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
            delta = self.field.sample_grid(grid, staggered=staggered) * dt
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


Inflow = lambda geometry, rate=1.0, batch_size=None: FieldEffect(ConstantField(geometry, rate), ('density',), GROW, tags=('inflow', 'effect'), batch_size=batch_size)
Fan = lambda geometry, acceleration, batch_size=None: FieldEffect(ConstantField(geometry, acceleration), ('velocity',), GROW, tags=('fan', 'effect'))
ConstantDensity = lambda geometry, density, batch_size=None: FieldEffect(ConstantField(geometry, density), ('density',), FIX)