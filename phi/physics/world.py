from .collective import CollectiveState, CollectivePhysics
from .smoke import *
from .burger import *


class StateTracker(object):

    def __init__(self, world, trajectorykey):
        self.world = world
        self.trajectorykey = trajectorykey

    @property
    def state(self):
        return self.world.state[self.trajectorykey]

    def __getattr__(self, item):
        return getattr(self.state, item)

    def __setattr__(self, key, value):
        if key in ('world', 'trajectorykey'):
            object.__setattr__(self, key, value)
        else:
            new_state = self.state.copied_with(**{key:value})
            self.world.state = self.world.state.state_replaced(self.state, new_state)



class World(object):

    def __init__(self):
        self._state = CollectiveState()
        self.physics = self._state.default_physics()
        self.observers = set()
        self.batch_size = None

        # Physics Shortcuts
        for target,source in {'Smoke': Smoke, 'Burger': Burger}.items():
            def wrapper(constructor):
                def buildadd(*args, **kwargs):
                    state = constructor(*args, **kwargs)
                    self.add(state)
                    return StateTracker(self, state.trajectorykey)
                return buildadd
            setattr(self, target, wrapper(source))

        # StaticObject Shortcuts
        for target, source in {'Inflow': Inflow, 'Obstacle': Obstacle}.items():
            def wrapper(constructor):
                def buildadd(*args, **kwargs):
                    state = constructor(*args, **kwargs)
                    self.add(state)
                    return StateTracker(self, state.trajectorykey)
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

    def step(self, state=None, dt=1.0):
        if state is None:
            self.physics.dt = dt
            self.state = self.physics.step(self._state, None, dt)
        else:
            if isinstance(state, StateTracker):
                state = state.state
            s = self.physics.substep(state, self._state, dt)
            self.state = self._state.state_replaced(state, s).copied_with(age=self._state.age + dt)

    def stepped(self, state=None, dt=1.0):
        if state is None:
            return self.physics.step(self._state, None, dt)
        else:
            if isinstance(state, StateTracker):
                state = state.state
            return self.physics.substep(state, self._state, dt)

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

    def add(self, state, physics=None):
        self.state += state
        if physics is not None:
            self.physics.add(state.trajectorykey, physics)

    def remove(self, state):
        self.state -= state
        self.physics.remove(state.trajectorykey)

    def clear(self):
        self._state = CollectiveState()
        self.physics = self._state.default_physics()
        self.state = self._state

    def reset(self):
        World.__init__(self)
        self.state = self._state



world = World()
