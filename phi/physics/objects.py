from .physics import *
from .material import *
from phi.math import shape
from copy import copy


class StaticObject(Physics):

    def __init__(self, objectstate):
        Physics.__init__(self)
        self.objectstate = objectstate

    def step(self, state):
        return state

    def shape(self, batch_size=1):
        return shape(self.objectstate)


class DynamicObject(Physics):

    def __init__(self):
        Physics.__init__(self)
        self.internal_time = 0.0

    def step(self, state):
        self.internal_time += self.dt
        result = self.object_at(self.internal_time)
        h = 1e-2 * self.dt
        perturbed = self.object_at(self.internal_time + h)
        result._velocity = (perturbed.geometry.center - result.geometry.center) / h
        return result

    def object_at(self, time):
        raise NotImplementedError(self)

    def shape(self, batch_size=1):
        return shape(self.object_at(-1.0))


class ObjectState(State):

    def __init__(self, geometry, velocity=0, tags=()):
        State.__init__(self, tags)
        self._geometry = geometry
        self._velocity = velocity

    @property
    def geometry(self):
        return self._geometry

    @property
    def velocity(self):
        return self._velocity

    def disassemble(self):
        return [], lambda _: copy(self)


class Obstacle(ObjectState):

    def __init__(self, geometry, material=SLIPPERY, velocity=0, tags=('obstacle',)):
        ObjectState.__init__(self, geometry=geometry, velocity=velocity, tags=tags)
        self.material = material


class Inflow(ObjectState):

    def __init__(self, geometry, rate=1.0, tags=('inflow',)):
        ObjectState.__init__(self, geometry=geometry, tags=tags)
        self.rate = rate



def geometries_with_tag(collectivestate, tag):
    return [o.geometry for o in collectivestate.states if tag in o.tags and isinstance(o, ObjectState)]