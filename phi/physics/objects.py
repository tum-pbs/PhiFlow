from .physics import *
from .world import *
from .material import *
from copy import copy


class StaticObject(Physics):

    def __init__(self, objectstate, world=world):
        Physics.__init__(self, world=world, state_tag=objectstate.tags[0], dt=0.0)
        self.objectstate = objectstate

    def step(self, state):
        return state

    def shape(self, batch_size=1):
        return shape(self.objectstate)


class DynamicObject(Physics):

    def __init__(self, world=world):
        Physics.__init__(self, world, state_tag=self.object_at(-1.0).tags[0])
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

    def __init__(self, geometry, material, velocity=0, tags=('obstacle',)):
        ObjectState.__init__(self, geometry=geometry, velocity=velocity, tags=tags)
        self.material = material


def obstacle(geometry, material=SLIPPERY, world=world):
    objectstate = Obstacle(geometry, material)
    staticobject = StaticObject(objectstate, world)
    world.add(staticobject)
    return staticobject


class Inflow(ObjectState):

    def __init__(self, geometry, rate, tags=('inflow',)):
        ObjectState.__init__(self, geometry=geometry, tags=tags)
        self.rate = rate


def inflow(geometry, rate=1.0, world=world):
    objectstate = Inflow(geometry, rate)
    staticobject = StaticObject(objectstate, world)
    world.add(staticobject)
    return staticobject


def geometries_with_tag(collectivestate, tag):
    return [o.geometry for o in collectivestate.states if tag in o.tags and isinstance(o, ObjectState)]