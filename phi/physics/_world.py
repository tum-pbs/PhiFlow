"""
Worlds are used to manage simulations consisting of multiple states.
A world uses a StateCollection with CollectivePhysics to resolve state dependencies.
Worlds also facilitate referencing states when performing a forward simulation.

A default World, called `world` is provided for convenience.
"""
import inspect
import warnings
from typing import TypeVar

from phi import geom
from phi.field import GeometryMask
from phi.physics._effect import Gravity
from ._physics import Physics, State, struct, _as_physics, Static


class StateProxy(object):
    """
    StateProxy mirrors all data_dict of a state in an associated world.
    While State objects are generally immutable, StateProxy also implements setting any attribute of the state.
    When an attribute is set, a copy of the state with the new value replaces the old state in the world.
    This object then mirrors the values of the new state.
    
    After world.step() is invoked, all Proxies of that world will mirror the state after stepping.
    
    To reference the current immutable state of

    Args:

    Returns:

    """

    def __init__(self, enclosing_world, state_name):
        self.world = enclosing_world
        self.state_name = state_name

    @property
    def state(self):
        """
        Finds and returns the state in the referenced world that matches the StateProxy's state_name.
        :return: State

        Args:

        Returns:

        """
        state = self.world.state[self.state_name]
        assert state is not None
        return state

    @state.setter
    def state(self, state):
        """
        Replaces the State in the referenced world.

        Args:
          state: 

        Returns:

        """
        assert state.name == self.state_name
        self.world.state = self.world.state.state_replaced(state)

    @property
    def physics(self):
        """
        Returns the Physics object used by the referenced world for the system.
        If not specified manually, the default physics is returned.

        Args:

        Returns:

        """
        physics = self.world.physics.for_(self.state)
        assert physics is not None
        return physics

    @physics.setter
    def physics(self, physics):
        """
        Sets a specific Physics object for the system in the referenced world.

        Args:
          physics: Physics

        Returns:

        """
        assert isinstance(physics, Physics)
        self.world.physics.add(self.state_name, physics)

    def step(self, dt=1.0, physics=None):
        """
        Steps this system in the referenced world. Other states are not affected.

        Args:
          dt: time increment (Default value = 1.0)
          physics: specific Physics object to use, defaults to self.physics

        Returns:

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

    Args:

    Returns:

    """

    def __init__(self, batch_size=None, add_default_objects=True):
        # --- Insert object / create proxy shortcuts ---
        self._state = self.physics = self.observers = self.batch_size = None
        self.reset(batch_size, add_default_objects)

    def reset(self, batch_size=None, add_default_objects=True):
        """
        Resets the world to the default configuration.
        This removes all States and observers.

        Args:
          batch_size: int or None (Default value = None)
          add_default_objects: if True, adds defaults like Gravity

        Returns:

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

        Args:

        Returns:

        """
        return self._state

    @property
    def age(self):
        """Alias for world.state.age"""
        return self._state.age

    @state.setter
    def state(self, state):
        """
        Sets the current state of the world and informs all observers.

        Args:
          state: StateCollection

        Returns:

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

        Args:
          state: State, StateProxy or None (Default value = None)
          dt: time increment (Default value = 1.0)
          physics: Physics object for the state or None for default

        Returns:
          evolved state if a specific state was provided

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

        Args:
          state:  (Default value = None)
          dt:  (Default value = 1.0)
          physics:  (Default value = None)

        Returns:

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

        Args:
          *states: 

        Returns:

        """
        warnings.warn('World.add_all() is deprecated. Use World.add(list_of_states) instead.', DeprecationWarning)
        for state in states:
            self.add(state)

    def remove(self, obj):
        """
        Remove a system or collection of systems from the world.

        Args:
          obj: one of the following: State, state name, subclass of State, tuple or list thereof

        Returns:

        """
        if inspect.isclass(obj):
            states = self.state.all_instances(obj)
            self.remove(states)
        elif isinstance(obj, (tuple, list)):
            for state in obj:
                self.remove(state)
        else:
            key = obj if isinstance(obj, str) else obj.name
            self.state = self.state.state_removed(key)
            self.physics.remove(key)

    def get_physics(self, state):
        """
        Looks up the Physics object associated with a given State or StateProxy.
        If no Physics object was registered manually, the state.default_physics() object is used.

        Args:
          state: State or StateProxy contained in this world

        Returns:
          Physics

        """
        if isinstance(state, StateProxy):
            state = state.state
        return self.physics.for_(state)

    def __getattr__(self, item):
        if item in self.state:
            return StateProxy(self, item)
        else:
            return object.__getattribute__(self, item)


def obstacle_mask(world_or_proxy):
    """
    Builds a binary Field, masking all obstacles in the world.

    Args:
      world_or_proxy: World or StateProxy object

    Returns:
      Field

    """
    world = world_or_proxy.world if isinstance(world_or_proxy, StateProxy) else world_or_proxy
    assert isinstance(world, World)
    geometries = [obstacle.geometry for obstacle in world.state.all_with_tag('obstacle')]
    return GeometryMask(geom.union(*geometries))


class StateCollection(dict):

    def __init__(self, states=None):
        """
        Create a state collection from a dictionary of states.

        Args:
          states(dict or list or tuple None): dict mapping from state names to states
        """
        if states is None:
            states = {}
        elif not isinstance(states, dict):
            states = {state.name: state for state in states}
        dict.__init__(self, states)

    def __setitem__(self, key, val):
        raise AttributeError('StateCollections are immutable')

    def all_with_tag(self, tag):
        return [s for s in self.values() if tag in s.tags]

    def all_instances(self, cls):
        return [s for s in self.values() if isinstance(s, cls)]

    def state_added(self, state):
        assert state.name not in self, 'A state with name "%s" is already present. Use state_replaced() to replace it.' % state.name
        new_states = self.copy()
        new_states[state.name] = state
        return StateCollection(new_states)

    def state_replaced(self, new_state):
        assert new_state.name in self, 'No state found with name "%s"' % new_state.name
        new_states = dict(self)
        new_states[new_state.name] = new_state
        return StateCollection(new_states)

    def state_removed(self, state):
        name = state if isinstance(state, str) else state.name
        new_states = dict(self)
        del new_states[name]
        return StateCollection(new_states)

    def find(self, name):
        warnings.warn("StateCollection.find is deprecated. Use statecollection[name] instead.", DeprecationWarning)
        return dict.__getitem__(self, name)

    def __getitem__(self, item):
        if isinstance(item, State):
            return self[item.name]
        if isinstance(item, str):
            return dict.__getitem__(self, item)
        if struct.isstruct(item):
            return struct.map(lambda x: self[x], item, content_type=struct.INVALID)
        try:
            return self[item.name]
        except AttributeError as e:
            pass
        raise ValueError('Illegal argument: %s' % item)

    def __getattr__(self, item):
        return self[item]

    def default_physics(self):
        warnings.warn("StateCollection will be removed in the future.", DeprecationWarning)
        return CollectivePhysics()

    def __repr__(self):
        return '[' + ', '.join((str(s) for s in self)) + ']'

    def __contains__(self, item):
        if isinstance(item, State):
            return item.name in self
        if isinstance(item, str):
            return dict.__contains__(self, item)
        raise ValueError('Illegal type: %s' % type(item))

    def __hash__(self):
        return 0

    @property
    def states(self):
        return self

    def copied_with(self, **kwargs):
        if len(kwargs) == 0:
            return self
        assert len(kwargs) == 1
        name, value = next(iter(kwargs.items()))
        assert name == 'states'
        return StateCollection(value)

    @property
    def shape(self):
        return StateCollection({name: state.shape for name, state in self.items()})

    @property
    def staticshape(self):
        return StateCollection({name: state.staticshape for name, state in self.items()})

    @property
    def dtype(self):
        return StateCollection({name: state.dtype for name, state in self.items()})


CollectiveState = StateCollection


class CollectivePhysics(Physics):

    def __init__(self):
        Physics.__init__(self, {})
        self.physics = {}  # map from name to Physics

    def step(self, state_collection, dt=1.0, **dependent_states):
        assert len(dependent_states) == 0
        if len(state_collection) == 0:
            return state_collection
        unhandled_states = list(state_collection.values())
        next_states = {}
        partial_next_state_collection = StateCollection(next_states)

        for sweep in range(len(state_collection)):
            for state in tuple(unhandled_states):
                physics = self.for_(state)
                if self._all_dependencies_fulfilled(physics.blocking_dependencies, state_collection, partial_next_state_collection):
                    next_state = self.substep(state, state_collection, dt, partial_next_state_collection=partial_next_state_collection)
                    assert next_state is not None, "step() called on %s returned None for state '%s'" % (type(physics).__name__, state)
                    assert isinstance(next_state, State), "step() called on %s dit not return a State but '%s' for state '%s'" % (type(physics).__name__, next_state, state)
                    assert next_state.name == state.name, "The state name must remain constant during step(). Caused by '%s' on state '%s'." % (type(physics).__name__, state)
                    next_states[next_state.name] = next_state
                    unhandled_states.remove(state)
            partial_next_state_collection = StateCollection(next_states)
            if len(unhandled_states) == 0:
                ordered_states = [partial_next_state_collection[state] for state in state_collection]
                return StateCollection(ordered_states)

        # Error
        errstr = 'Cyclic blocking_dependencies in simulation: %s' % unhandled_states
        for state in tuple(unhandled_states):
            physics = self.for_(state)
            state_dict = self._gather_dependencies(physics.blocking_dependencies, state_collection, {})
            errstr += '\nState "%s" with physics "%s" depends on %s' % (state, physics, state_dict)
        raise AssertionError(errstr)

    def substep(self, state, state_collection, dt, override_physics=None, partial_next_state_collection=None):
        physics = self.for_(state) if override_physics is None else override_physics
        # --- gather dependencies
        dependent_states = {}
        self._gather_dependencies(physics.dependencies, state_collection, dependent_states)
        if partial_next_state_collection is not None:
            self._gather_dependencies(physics.blocking_dependencies, partial_next_state_collection, dependent_states)
        # --- execute step ---
        next_state = physics.step(state, dt, **dependent_states)
        return next_state

    def _gather_dependencies(self, dependencies, state_collection, result_dict):
        for statedependency in dependencies:
            if statedependency.state_name is not None:
                matching_states = state_collection.find(statedependency.state_name)
            else:
                matching_states = state_collection.all_with_tag(statedependency.tag)
            if statedependency.single_state:
                assert len(matching_states) == 1, 'Dependency %s requires 1 state but found %d' % (statedependency, len(matching_states))
                value = matching_states[0]
            else:
                value = tuple(matching_states)
            result_dict[statedependency.parameter_name] = value
        return result_dict

    def _all_dependencies_fulfilled(self, dependencies, all_states, computed_states):
        state_dict = self._gather_dependencies(dependencies, all_states, {})
        for name, states in state_dict.items():
            if isinstance(states, tuple):
                for state in states:
                    if state.name not in computed_states:
                        return False
            else:  # single state
                if states.name not in computed_states:
                    return False
        return True

    def for_(self, state):
        return self.physics[state.name] if state.name in self.physics else state.default_physics()

    def add(self, name, physics):
        self.physics[name] = _as_physics(physics)

    def remove(self, name):
        if name in self.physics:
            del self.physics[name]


world = World()
