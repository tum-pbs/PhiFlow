from phi import struct
from phi.geom.geometry import Geometry

from .field import GeometryMask
from .field.effect import FieldEffect
from .material import CLOSED, Material
from .physics import Physics, State


@struct.definition()
class Obstacle(State):

    def __init__(self, geometry, material=CLOSED, velocity=0, tags=('obstacle',), **kwargs):
        State.__init__(self, **struct.kwargs(locals()))

    @struct.constant()
    def geometry(self, geometry):
        assert isinstance(geometry, Geometry)
        return geometry

    @struct.constant(default=CLOSED)
    def material(self, material):
        assert isinstance(material, Material)
        return material

    @struct.constant(default=0)
    def velocity(self, velocity):
        return velocity

    @struct.constant(default=0)
    def angular_velocity(self, av):
        return av

    @struct.derived()
    def is_stationary(self):
        return self.velocity is 0 and self.angular_velocity is 0


class GeometryMovement(Physics):

    def __init__(self, geometry_function):
        Physics.__init__(self)
        self.geometry_at = geometry_function

    def step(self, obj, dt=1.0, **dependent_states):
        next_geometry = self.geometry_at(obj.age + dt)
        h = 1e-2 * dt if dt > 0 else 1e-2
        perturbed_geometry = self.geometry_at(obj.age + dt + h)
        velocity = (perturbed_geometry.center - next_geometry.center) / h
        if isinstance(obj, Obstacle):
            return obj.copied_with(geometry=next_geometry, velocity=velocity, age=obj.age + dt)
        if isinstance(obj, FieldEffect):
            with struct.ALL_ITEMS:
                next_field = struct.map(lambda x: x.copied_with(geometries=next_geometry) if isinstance(x, GeometryMask) else x, obj.field, leaf_condition=lambda x: isinstance(x, GeometryMask))
            return obj.copied_with(field=next_field, age=obj.age + dt)
