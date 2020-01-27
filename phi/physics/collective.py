import six
from phi.struct.context import skip_validate

from .physics import Physics, State, struct


@struct.definition()
class StateCollection(struct.Struct):

    def __init__(self, states=None, **kwargs):
        struct.Struct.__init__(self, **struct.kwargs(locals()))

    @struct.variable()
    def states(self, states):
        if states is None:
            return {}
        if isinstance(states, (tuple, list)):
            return {state.name: state for state in states}
        assert isinstance(states, dict)
        return states.copy()

    def all_with_tag(self, tag):
        return [s for s in self.states.values() if tag in s.tags]

    def all_instances(self, cls):
        return [s for s in self.states.values() if isinstance(s, cls)]

    def state_added(self, state):
        assert state.name not in self.states,\
            'A state with name "%s" is already present. Use state_replaced() to replace it.' % state.name
        new_states = self.states.copy()
        new_states[state.name] = state
        return self.copied_with(states=new_states)

    def state_replaced(self, new_state):
        assert new_state.name in self.states, 'No state found with name "%s"' % new_state.name
        new_states = {state.name: (state if state.name != new_state.name else new_state) for state in self.states.values()}
        return self.copied_with(states=new_states)

    def state_removed(self, state):
        name = state if isinstance(state, six.string_types) else state.name
        new_states = self.states.copy()
        del new_states[name]
        return self.copied_with(states=new_states)

    def find(self, name):
        return self.states[name]

    def __getitem__(self, item):
        if isinstance(item, State):
            return self[item.name]
        if isinstance(item, six.string_types):
            return self.find(item)
        if struct.isstruct(item):
            with struct.unsafe():
                return struct.map(lambda x: self[x], item)
        try:
            return self[item.name]
        except AttributeError as e:
            pass
        raise ValueError('Illegal argument: %s' % item)

    def __getattr__(self, item):
        if item.startswith('_'):
            return struct.Struct.__getattribute__(self, item)
        if item in self.states:
            return self.states[item]
        return struct.Struct.__getattribute__(self, item)

    def default_physics(self):
        return CollectivePhysics()

    def __repr__(self):
        return '[' + ', '.join((str(s) for s in self.states)) + ']'

    def __len__(self):
        return len(self.states)

    def __contains__(self, item):
        if isinstance(item, State):
            return item.name in self.states
        if isinstance(item, six.string_types):
            return item in self.states
        raise ValueError('Illegal type: %s' % type(item))

    def _set_items(self, **kwargs):
        for name, value in kwargs.items():
            if name in ('states', 'age'):
                getattr(self.__class__, name).set(self, value)
            else:
                self._states = self.states.copy()
                if not skip_validate():
                    assert isinstance(value, State)
                    assert value.name == name, 'Inconsisten names: trying to assign state "%s" to name "%s"' % (value.name, name)
                    assert 'states' not in kwargs
                self.states[name] = value
        return self

    def __to_dict__(self, item_condition=None):
        return self.states.copy()

    def __properties__(self):
        return {}

    def __properties_dict__(self):
        result = {}
        for state in self.states.values():
            result[state.name] = struct.properties_dict(state)
        result['type'] = str(self.__class__.__name__)
        result['module'] = str(self.__class__.__module__)
        return result

    @property
    def shape(self):
        return struct.map(lambda state: state.shape, self, recursive=False)


CollectiveState = StateCollection


class CollectivePhysics(Physics):

    def __init__(self):
        Physics.__init__(self, {})
        self.physics = {}  # map from name to Physics

    def step(self, state_collection, dt=1.0, **dependent_states):
        assert len(dependent_states) == 0
        if len(state_collection) == 0:
            return state_collection
        unhandled_states = list(state_collection.states.values())
        next_states = {}
        partial_next_state_collection = StateCollection(next_states)

        for sweep in range(len(state_collection)):
            for state in tuple(unhandled_states):
                physics = self.for_(state)
                if self._all_dependencies_fulfilled(physics.blocking_dependencies, state_collection, partial_next_state_collection):
                    next_state = self.substep(state, state_collection, dt, partial_next_state_collection=partial_next_state_collection)
                    assert next_state.name == state.name, "The state name must remain constant during step(). Caused by '%s' on state '%s'." % (type(physics).__name__, state)
                    next_states[next_state.name] = next_state
                    unhandled_states.remove(state)
            partial_next_state_collection = StateCollection(next_states)
            if len(unhandled_states) == 0:
                ordered_states = [partial_next_state_collection[state] for state in state_collection.states]
                return partial_next_state_collection.copied_with(states=ordered_states)

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
        self.physics[name] = physics

    def remove(self, name):
        if name in self.physics:
            del self.physics[name]
