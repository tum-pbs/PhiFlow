"""
Worlds are used to manage simulations consisting of multiple states.
A world uses a StateCollection with CollectivePhysics to resolve state dependencies.
Worlds also facilitate referencing states when performing a forward simulation.

A default World, called `world` is provided for convenience.
"""
import inspect
import warnings
from typing import TypeVar

import six

from .collective import StateCollection
from .field.effect import Gravity
from .physics import Physics, State, Static


class StateProxy(object):
    """
    StateProxy mirrors all data_dict of a state in an associated world.
    While State objects are generally immutable, StateProxy also implements setting any attribute of the state.
    When an attribute is set, a copy of the state with the new value replaces the old state in the world.
    This object then mirrors the values of the new state.

    After world.step() is invoked, all Proxies of that world will mirror the state after stepping.

    To reference the current immutable state of
    """

    def __init__(self, enclosing_world, state_name):
        self.world = enclosing_world
        self.state_name = state_name

    @property
    def state(self):
        """
Finds and returns the state in the referenced world that matches the StateProxy's state_name.
        :return: State
        """
        state = self.world.state[self.state_name]
        assert state is not None
        return state

    @state.setter
    def state(self, state):
        """
Replaces the State in the referenced world.
        """
        assert state.name == self.state_name
        self.world.state = self.world.state.state_replaced(state)

    @property
    def physics(self):
        """
Returns the Physics object used by the referenced world for the system.
If not specified manually, the default physics is returned.
        """
        physics = self.world.physics.for_(self.state)
        assert physics is not None
        return physics

    @physics.setter
    def physics(self, physics):
        """
Sets a specific Physics object for the system in the referenced world.
        :param physics: Physics
        """
        assert isinstance(physics, Physics)
        self.world.physics.add(self.state_name, physics)

    def step(self, dt=1.0, physics=None):
        """
Steps this system in the referenced world. Other states are not affected.
        :param dt: time increment
        :param physics: specific Physics object to use, defaults to self.physics
        """
        self.world.step(self, dt=dt, physics=physics)

    def __getattr__(self, item):
        """
Returns an attribute from the referenced State.
        :param item: attribute name
        """
        assert item not in ('world', 'state_name', 'physics', 'state')
        return getattr(self.state, item)

    def __setattr__(self, key, value):
        """
Changes the referenced state by replacing it in the referenced world.
        :param key: State attribute name
        :param value: new value
        """
        if key in ('world', 'state_name', 'physics', 'state'):
            object.__setattr__(self, key, value)
        else:
            self.state = self.state.copied_with(**{key:value})


# pylint: disable-msg = invalid-name
S = TypeVar('S', bound=State)


class World(object):
    """
    A world object defines a global state as well as a set of rules (Physics objects) that definition how the state evolves.

    The world manages dependencies among the contained simulations and provides convenience methods for creating proxies for specific simulations.

    The method world.step() evolves the whole state or optionally a specific state in time.
    """

    def __init__(self, batch_size=None, add_default_objects=True):
        # --- Insert object / create proxy shortcuts ---
        self._state = self.physics = self.observers = self.batch_size = None
        self.reset(batch_size, add_default_objects)

    def reset(self, batch_size=None, add_default_objects=True):
        """
Resets the world to the default configuration.
This removes all States and observers.
        :param batch_size: int or None
        :param add_default_objects: if True, adds defaults like Gravity
        """
        self._state = StateCollection()
        self.physics = self._state.default_physics()
        self.observers = set()
        self.batch_size = batch_size
        if add_default_objects:
            self.add(Gravity())

    @property
    def state(self):
        """
Returns the current state of the world.
        :return: StateCollection
        """
        return self._state

    @property
    def age(self):
        """
Alias for world.state.age
        """
        return self._state.age

    @state.setter
    def state(self, state):
        """
Sets the current state of the world and informs all observers.
        :param state: StateCollection
        """
        assert state is not None
        assert isinstance(state, StateCollection)
        self._state = state
        for observer in self.observers:
            observer(self)

    def step(self, state=None, dt=1.0, physics=None):
        """
        Evolves the current world state by a time increment dt.
        If state is provided, only that state is evolved, leaving the others untouched.
        The optional physics parameter can then be used to override the default physics.
        Otherwise, all states are evolved.

        Calling World.step resolves all dependencies among simulations and then calls Physics.step on each simulation to evolve the states.

        Invoking this method alters the world state. To to_field a copy of the state, use :func:`World.stepped <~world.World.stepped>` instead.
            :param state: State, StateProxy or None
            :param dt: time increment
            :param physics: Physics object for the state or None for default
            :return: evolved state if a specific state was provided
        """
        if state is None:
            if physics is None:
                physics = self.physics
            self.state = physics.step(self._state, dt)
            return self.state
        else:
            if isinstance(state, StateProxy):
                state = state.state
            s = self.physics.substep(state, self._state, dt, override_physics=physics)
            self.state = self._state.state_replaced(s)
            return s

    def stepped(self, state=None, dt=1.0, physics=None):
        """
Similar to step() but does not change the state of the world. Instead, the new state is returned.
        """
        if state is None:
            if physics is None:
                physics = self.physics
            return physics.step(self._state, None, dt)
        else:
            if isinstance(state, StateProxy):
                state = state.state
            return self.physics.substep(state, self._state, dt, override_physics=physics)

    def add(self, state, physics=None):
        # type: (S, Physics) -> S
        """
Adds a State to the world that will be stepped forward in time each time world.step() is invoked.
        :param state: State or list of States
        :param physics: (optional) Physics to use during world.step(). If a list was provided for `state`, a matching list must be given for `state`.
        :return: StateProxy referencing the current state of the added system. If world.state is updated (e.g. because world.step() was called), the StateProxy will refer to the updated values.
        """
        if isinstance(state, dict):
            raise ValueError('Cannot add dict to world. Maybe you meant world.add(**dict)?')
        if isinstance(state, (tuple, list)):
            assert isinstance(physics, (tuple, list))
            assert len(state) == len(physics)
            return [self.add(s, p) for s, p in zip(state, physics)]
        else:
            if physics is not None:
                self.physics.add(state.name, physics)
            elif state.default_physics() is not None and not isinstance(state.default_physics(), Static):
                warnings.warn('No physics provided to world.add(%s). In the future this will default to static physics' % state)
            self.state = self.state.state_added(state)
            return StateProxy(self, state.name)

    def add_all(self, *states):
        """
Add a collection of states to the system using world.add(state).
        """
        warnings.warn('World.add_all() is deprecated. Use World.add(list_of_states) instead.', DeprecationWarning)
        for state in states:
            self.add(state)

    def remove(self, obj):
        """
Remove a system or collection of systems from the world.
        :param obj: one of the following: State, state name, subclass of State, tuple or list thereof
        """
        if inspect.isclass(obj):
            states = self.state.all_instances(obj)
            self.remove(states)
        elif isinstance(obj, (tuple, list)):
            for state in obj:
                self.remove(state)
        else:
            key = obj if isinstance(obj, six.string_types) else obj.name
            self.state = self.state.state_removed(key)
            self.physics.remove(key)

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

    def __getattr__(self, item):
        if item in self.state:
            return StateProxy(self, item)
        else:
            return object.__getattribute__(self, item)


world = World()
