from typing import TypeVar
import inspect

from .physics import State, Physics, TrajectoryKey
from .collective import CollectiveState
from phi.physics.field.effect import Gravity


class StateProxy(object):
    """
    StateProxy mirrors all attributes of a state in an associated world.
    While State objects are generally immutable, StateProxy also implements setting any attribute of the state.
    When an attribute is set, a copy of the state with the new value replaces the old state in the world.
    This object then mirrors the values of the new state.

    After world.step() is invoked, all Proxies of that world will mirror the state after stepping.

    To reference the current immutable state of
    """

    def __init__(self, world, trajectorykey):
        self.world = world
        self.trajectorykey = trajectorykey

    @property
    def state(self):
        state = self.world.state[self.trajectorykey]
        assert state is not None
        return state

    @state.setter
    def state(self, state):
        self.world.state = self.world.state.state_replaced(self.state, state)

    @property
    def physics(self):
        physics = self.world.physics.for_(self.state)
        assert physics is not None
        return physics

    @physics.setter
    def physics(self, phys):
        self.world.physics.add(self.trajectorykey, phys)

    def step(self, dt=1.0, physics=None):
        self.world.step(self, dt=dt, physics=physics)

    def __getattr__(self, item):
        assert item not in ('world', 'trajectorykey', 'physics', 'state')
        return getattr(self.state, item)

    def __setattr__(self, key, value):
        if key in ('world', 'trajectorykey', 'physics', 'state'):
            object.__setattr__(self, key, value)
        else:
            self.state = self.state.copied_with(**{key:value})


S = TypeVar('S', bound=State)


class World(object):
    """
    A world object defines a global state as well as a set of rules (Physics objects) that define how the state evolves.

    The world manages dependencies among the contained simulations and provides convenience methods for creating proxies for specific simulations.

    The method world.step() evolves the whole state or optionally a specific state in time.
    """

    def __init__(self, batch_size=None, add_default_objects=True):
        # --- Insert object / create proxy shortcuts ---
        self.reset(batch_size, add_default_objects)

    def reset(self, batch_size=None, add_default_objects=True):
        self._state = CollectiveState()
        self.physics = self._state.default_physics()
        self.observers = set()
        self.batch_size = batch_size
        if add_default_objects:
            self.add(Gravity())

    @property
    def state(self):
        return self._state

    @property
    def age(self):
        return self._state.age

    @state.setter
    def state(self, state):
        assert state is not None
        self._state = state
        for observer in self.observers: observer(self)

    def step(self, state=None, dt=1.0, physics=None):
        """
        Evolves the current world state by a time increment dt.
        If state is provided, only that state is evolved, leaving the others untouched.
        The optional physics parameter can then be used to override the default physics.
        Otherwise, all states are evolved.

        Calling World.step resolves all dependencies among simulations and then calls Physics.step on each simulation to evolve the states.

        Invoking this method alters the world state. To to_field a copy of the state, use :func:`World.stepped <~world.World.stepped>` instead.
            :param state: State, StateProxy or None
            :param dt: time increment, default 1.0
            :param physics: Physics object for the state or None for default
            :return: evolved state if a specific state was provided
        """
        if state is None:
            if physics is None: physics = self.physics
            self.state = physics.step(self._state, dt)
        else:
            if isinstance(state, StateProxy):
                state = state.state
            s = self.physics.substep(state, self._state, dt, override_physics=physics)
            self.state = self._state.state_replaced(state, s).copied_with(age=self._state.age + dt)
            return s

    def stepped(self, state=None, dt=1.0, physics=None):
        if state is None:
            if physics is None: physics = self.physics
            return physics.step(self._state, None, dt)
        else:
            if isinstance(state, StateProxy):
                state = state.state
            return self.physics.substep(state, self._state, dt, override_physics=physics)

    def add(self, state, physics=None):
        # type: (S, Physics) -> S
        """
        Adds a State to world.state and creates a StateProxy for that state.

        :param state: State object to add
        :param physics: Physics to use for stepping or None for state.default_physics()
        :return: a StateProxy representing the added state. If world.state is updated (e.g. because world.step() was called), the StateProxy will refer to the updated values.
        """
        self.state += state
        if state._batch_size is None:
            state._batch_size = self.batch_size
        if physics is not None:
            self.physics.add(state.trajectorykey, physics)
        return StateProxy(self, state.trajectorykey)

    def add_all(self, *states):
        self.state += states

    def remove(self, obj):
        if inspect.isclass(obj):
            states = self.state.all_instances(obj)
            return self.remove(states)
        elif isinstance(obj, (tuple,list)):
            for state in obj:
                self.remove(state)
        else:
            key = obj if isinstance(obj, TrajectoryKey) else obj.trajectorykey
            self.state = self.state.trajectory_removed(key)
            self.physics.remove(key)

    def clear(self):
        self._state = CollectiveState()
        self.physics = self._state.default_physics()
        self.state = self._state

    def get_physics(self, state):
        """
        Looks up the Physics object associated with a given State or StateProxy.
        If no Physics object was registered manually, the state.default_physics() object is used.
        
        :param state: State or StateProxy contained in this world
        :return: Physics
        """
        if isinstance(state, StateProxy):
            state = state.state
        return self.physics.for_(state)


world = World()
