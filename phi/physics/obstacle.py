from .physics import *
from .material import *
from .effect import *


class Obstacle(State):
    __struct__ = State.__struct__.extend((), ('_geometry', '_material', '_velocity'))

    def __init__(self, geometry, material=SLIPPERY, velocity=0, tags=('obstacle',), age=0.0, batch_size=None):
        State.__init__(self, tags=tags, age=age, batch_size=batch_size)
        self._material = material
        self._geometry = geometry
        self._velocity = velocity

    @property
    def geometry(self):
        return self._geometry

    @property
    def material(self):
        return self._material

    @property
    def velocity(self):
        return self._velocity


class GeometryMovement(Physics):

    def __init__(self, geometry_function):
        Physics.__init__(self, {})
        self.geometry_at = geometry_function

    def step(self, obj, dt=1.0, **dependent_states):
        next_geometry = self.geometry_at(obj.age + dt)
        h = 1e-2 * dt if dt > 0 else 1e-2
        perturbed_geometry = self.geometry_at(obj.age + dt + h)
        velocity = (perturbed_geometry.center - next_geometry.center) / h
        if isinstance(obj, Obstacle):
            return obj.copied_with(geometry=next_geometry, velocity=velocity, age=obj.age + dt)
        if isinstance(obj, FieldEffect):
            return obj.copied_with(field=obj.field.copied_with(bounds=next_geometry), age=obj.age + dt)