from sets import ImmutableSet

from phi import math
from phi.physics.material import SLIPPERY


class WorldState(object):

    def __init__(self, objects=()):
        self._objects = objects if isinstance(objects, ImmutableSet) else ImmutableSet(objects)

    def __add__(self, other):
        if isinstance(other, WorldObject):
            return WorldState(self._objects | ImmutableSet((other,)))
        if isinstance(other, WorldState):
            return WorldState(self._objects | other._objects)
        if isinstance(other, (tuple, list)):
            return WorldState(self._objects | ImmutableSet(other))

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, WorldObject):
            return WorldState(self._objects - ImmutableSet((other,)))
        if isinstance(other, WorldState):
            return WorldState(self._objects - other._objects)
        if isinstance(other, (tuple, list)):
            return WorldState(self._objects - ImmutableSet(other))

    def objects_with_tag(self, tag):
        return filter(lambda o: tag in o.tags, self._objects)

    def geometries_with_tag(self, tag):
        return [o.geometry for o in self.objects_with_tag(tag)]

    @property
    def objects(self):
        return self._objects



class World(object):

    def __init__(self):
        self._state = WorldState()
        self._simulations = []
        self._observers = set()

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state
        for observer in self._observers: observer(self)

    def add(self, objects):
        self.state += objects
        return objects

    def remove(self, objects):
        self.state -= objects

    def on_change(self, observer):
        """
Register an observer that will be called when objects are added to the world or removed from the world.
The observer must define __call__ and will be given the world as parameter.
        :param observer:
        """
        self._observers.add(observer)

    def remove_on_change(self, observer):
        self._observers.remove(observer)

    def register_simulation(self, simulation):
        self._simulations.append(simulation)

    def unregister_simulation(self, simulation):
        self._simulations.remove(simulation)



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
    return world.add(Obstacle(geometry, material))


class Inflow(WorldObject):

    def __init__(self, geometry, rate, tags=('inflow')):
        WorldObject.__init__(self, geometry, tags)
        self.rate = rate


def inflow(geometry, rate=1.0, world=world):
    return world.add(Inflow(geometry, rate))


def inflow_mask(world, grid):
    inflows = world.state.objects_with_tag('inflow')
    if len(inflows) == 0:
        return grid.zeros()
    location = grid.center_points()
    return math.add([inflow.geometry.value_at(location) * inflow.rate for inflow in inflows])


def geometry_mask(world, grid, tag):
    geometries = world.state.geometries_with_tag(tag)
    if len(geometries) == 0:
        return grid.zeros()
    location = grid.center_points()
    return math.max([geometry.value_at(location) for geometry in geometries], axis=0)