import warnings
import six

from .physics import Physics, State, struct, _ChainedPhysics, _as_physics


class StateCollection(dict):

    def __init__(self, states=None):
        """
Create a state collection from a dictionary of states.
        :param states: dict mapping from state names to states
        :type states: dict or list or tuple None
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
        name = state if isinstance(state, six.string_types) else state.name
        new_states = dict(self)
        del new_states[name]
        return StateCollection(new_states)

    def find(self, name):
        warnings.warn("StateCollection.find is deprecated. Use statecollection[name] instead.", DeprecationWarning)
        return dict.__getitem__(self, name)

    def __getitem__(self, item):
        if isinstance(item, State):
            return self[item.name]
        if isinstance(item, six.string_types):
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
        if isinstance(item, six.string_types):
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
