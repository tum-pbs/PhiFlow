from .collective import CollectiveState, CollectivePhysics
from .smoke import *
from .burger import *


class World(object):

    def __init__(self):
        self.physics = CollectivePhysics()
        self._state = self.physics.initial_state()
        self.observers = set()
        self.batch_size = None

        # Physics Shortcuts
        for target,source in {'Smoke': Smoke, 'Burger': Burger}.items():
            def wrapper(sourcefunction):
                def buildadd(*args, **kwargs):
                    obj = sourcefunction(*args, **kwargs)
                    self.add(obj)
                    return obj
                return buildadd
            setattr(self, target, wrapper(source))

        # StaticObject Shortcuts
        for target, source in {'Inflow': Inflow, 'Obstacle': Obstacle}.items():
            def wrapper(sourcefunction):
                def buildadd(*args, **kwargs):
                    obj = sourcefunction(*args, **kwargs)
                    self.add(StaticObject(obj))
                    return obj
                return buildadd

            setattr(self, target, wrapper(source))

    Smoke = Smoke
    Burger = Burger

    Inflow = Inflow
    Obstacle = Obstacle

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        assert state is not None
        self._state = state
        for observer in self.observers: observer(self)

    def step(self, dt=1.0):
        self.physics.dt = dt
        self.state = self.physics.step(self._state)

#     def on_change(self, observer):
#         """
# Register an observer that will be called when states are added to the world or removed from the world.
# The observer must define __call__ and will be given the world as parameter.
#         :param observer:
#         """
#         self.observers.add(observer)
#
#     def remove_on_change(self, observer):
#         self.observers.remove(observer)

    def add(self, physics, initial_state=None):
        self.physics.add(physics)
        if initial_state is None:
            initial_state = physics.initial_state(batch_size=self.batch_size)
        self.state += initial_state
        return physics

    def remove(self, physics, remove_states=True):
        self.physics.remove(physics)
        if remove_states:
            states = self.state.get_by_tag(physics.state_tag)
            self.state -= states


world = World()
