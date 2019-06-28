from .physics import *
from .material import *
from phi.math import shape
from copy import copy


class GeometryMovement(Physics):

    def __init__(self, geometry_function):
        Physics.__init__(self, {})
        self.geometry_at = geometry_function

    def step(self, obj, dt=1.0, **dependent_states):
        next_geometry = self.geometry_at(obj.age + dt)
        h = 1e-2 * dt if dt > 0 else 1e-2
        perturbed_geometry = self.geometry_at(obj.age + dt + h)
        velocity = (perturbed_geometry.center - next_geometry.center) / h
        return obj.copied_with(geometry=next_geometry, velocity=velocity, age=obj.age + dt)



class ObjectState(State):
    __struct__ = State.__struct__.extend((), ('_geometry', '_velocity'))

    def __init__(self, geometry, velocity=0, tags=(), batch_size=None):
        State.__init__(self, tags=tags, batch_size=batch_size)
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
    __struct__ = ObjectState.__struct__.extend((), ('_material',))

    def __init__(self, geometry, material=SLIPPERY, velocity=0, tags=('obstacle',), batch_size=None):
        ObjectState.__init__(self, geometry=geometry, velocity=velocity, tags=tags, batch_size=batch_size)
        self._material = material

    @property
    def material(self):
        return self._material


class Inflow(ObjectState):
    __struct__ = ObjectState.__struct__.extend((), ('_rate',))

    def __init__(self, geometry, rate=1.0, tags=('inflow',), batch_size=None):
        ObjectState.__init__(self, geometry=geometry, tags=tags, batch_size=batch_size)
        self._rate = rate

    @property
    def rate(self):
        return self._rate



def geometries_with_tag(collectivestate, tag):
    return [o.geometry for o in collectivestate.states if tag in o.tags and isinstance(o, ObjectState)]