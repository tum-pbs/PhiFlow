from phi import math
from phi.physics.material import SLIPPERY


class World(object):

    def __init__(self):
        self._objects = set()
        self._observers = set()


    def add_object(self, object):
        self._objects.add(object)
        for observer in self._observers: observer(self, [object])
        return object

    def remove_object(self, object):
        self._objects.remove(object)
        for observer in self._observers: observer(self, [object])

    def on_change(self, observer):
        """
Register an observer that will be called when objects are added to the world or removed from the world.
The observer must define __call__ and will be given the world and a list of changed objects as parameters.
        :param observer:
        """
        self._observers.add(observer)

    def remove_on_change(self, observer):
        self._observers.remove(observer)

    def objects_with_tag(self, tag):
        return filter(lambda o: tag in o.tags, self._objects)

    def geometries_with_tag(self, tag):
        return [o.geometry for o in self.objects_with_tag(tag)]

    @property
    def all_objects(self):
        return self._objects



world = World()


class WorldObject(object):

    def __init__(self, geometry, tags=()):
        self.geometry = geometry
        self.tags = tags


class Obstacle(WorldObject):

    def __init__(self, geometry, material, tags=('obstacle',)):
        WorldObject.__init__(self, geometry, tags)
        self.material = material


def obstacle(geometry, material=SLIPPERY, world=world):
    return world.add_object(Obstacle(geometry, material))


class Inflow(WorldObject):

    def __init__(self, geometry, rate, tags=('inflow')):
        WorldObject.__init__(self, geometry, tags)
        self.rate = rate


def inflow(geometry, rate=1.0, world=world):
    return world.add_object(Inflow(geometry, rate))


def inflow_mask(world, grid):
    location = grid.center_points()
    inflows = world.objects_with_tag('inflow')
    return math.add([inflow.geometry.value_at(location) * inflow.rate for inflow in inflows])


def geometry_mask(world, grid, tag):
    location = grid.center_points()
    geometries = world.geometries_with_tag(tag)
    return math.max([geometry.value_at(location) for geometry in geometries], axis=0)