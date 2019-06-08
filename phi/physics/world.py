from .collective import CollectiveState, CollectivePhysics
from phi.math import zeros


class World(object):

    def __init__(self):
        self._state = CollectiveState()
        self.observers = set()
        self.batch_size = None
        self.physics = CollectivePhysics(self)

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

    def on_change(self, observer):
        """
Register an observer that will be called when states are added to the world or removed from the world.
The observer must define __call__ and will be given the world as parameter.
        :param observer:
        """
        self.observers.add(observer)

    def remove_on_change(self, observer):
        self.observers.remove(observer)

    def add(self, physics, initial_state=None):
        self.physics.add(physics)
        if initial_state is None:
            initial_state = zeros(physics.shape())
        assert physics.state_tag in initial_state.tags, "Tag %s missing from accompanying state" % physics.state_tag
        self.state += initial_state

    def remove(self, physics, remove_states=True):
        self.physics.remove(physics)
        if remove_states:
            states = self.state.get_by_tag(physics.state_tag)
            self.state -= states



world = World()
