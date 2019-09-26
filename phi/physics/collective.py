from .physics import *
from phi.math import *
import six


class CollectiveState(State):
    __struct__ = State.__struct__.extend(('_states',))

    def __init__(self, states=(), age=0.0):
        State.__init__(self, tags=(), age=age)
        self._states = states if isinstance(states, tuple) else tuple(states)

    def __add__(self, other):
        if isinstance(other, CollectiveState):
            return CollectiveState(self._states + other._states)
        if isinstance(other, State):
            return CollectiveState(self._states + (other,))
        if isinstance(other, (tuple, list)):
            return CollectiveState(self._states + tuple(other))
        raise ValueError("Illegal operation: CollectiveState + %s" % type(other))

    __radd__ = __add__

    def get_by_tag(self, tag):
        return [o for o in self._states if tag in o.tags]

    @property
    def states(self):
        return self._states

    def __names__(self):
        return [None for state in self._states]

    def __values__(self):
        return self.states

    def state_replaced(self, old_state, new_state):
        new_states = tuple(map(lambda s: new_state if s == old_state else s, self._states))
        return self.copied_with(states=new_states)

    def __getitem__(self, item):
        if isinstance(item, State):
            if item in self._states: return item
            else: return self[item.trajectorykey]
        if isinstance(item, TrajectoryKey):
            states = list(filter(lambda s: s.trajectorykey==item, self._states))
            assert len(states) == 1, 'CollectiveState[%s] returned %d states. All contents: %s' % (item, len(states), self.states)
            return states[0]
        if isinstance(item, six.string_types):
            return self.get_by_tag(item)
        if isinstance(item, (tuple, list)):
            return [self[i] for i in item]
        try:
            return self[item.trajectorykey]
        except AttributeError as e:
            pass
        raise ValueError('Illegal argument: %s' % item)

    def default_physics(self):
        phys = CollectivePhysics()
        for state in self._states:
            phys.add(state.trajectorykey, state.default_physics())
        return phys

    def __repr__(self):
        return '[' + ', '.join((str(s) for s in self._states)) + ']'

    def __len__(self):
        return len(self.states)

    def __contains__(self, item):
        if isinstance(item, State):
            return item in self._states
        if isinstance(item, TrajectoryKey):
            for state in self._states:
                if state.trajectorykey == item: return True
            return False
        raise ValueError('Illegal type: %s' % type(item))


class CollectivePhysics(Physics):

    def __init__(self):
        Physics.__init__(self, {})
        self._physics = {}  # map from TrajectoryKey to Physics

    def step(self, collectivestate, dt=1.0, **dependent_states):
        assert len(dependent_states) == 0
        if len(collectivestate) == 0: return CollectiveState(age=collectivestate.age + dt)
        unhandled_states = list(collectivestate.states)
        next_states = []
        partial_next_collectivestate = CollectiveState(next_states, age=collectivestate.age + dt)

        for sweep in range(len(collectivestate)):
            for state in tuple(unhandled_states):
                physics = self.for_(state)
                if self._all_dependencies_fulfilled(physics.blocking_dependencies, collectivestate, partial_next_collectivestate):
                    next_state = self.substep(state, collectivestate, dt, partial_next_collectivestate=partial_next_collectivestate)
                    next_states.append(next_state)
                    unhandled_states.remove(state)
            partial_next_collectivestate = CollectiveState(next_states, age=collectivestate.age + dt)
            if len(unhandled_states) == 0:
                ordered_states = [partial_next_collectivestate[state] for state in collectivestate.states]
                return partial_next_collectivestate.copied_with(states=ordered_states)

        raise AssertionError('Cyclic blocking_dependencies in simulation: %s' % unhandled_states)

    def substep(self, state, collectivestate, dt, override_physics=None, partial_next_collectivestate=None):
        physics = self.for_(state) if override_physics is None else override_physics
        # --- gather dependencies
        dependent_states = {}
        self._gather_dependencies(physics.dependencies, collectivestate, dependent_states)
        if partial_next_collectivestate is not None:
            self._gather_dependencies(physics.blocking_dependencies, partial_next_collectivestate, dependent_states)
        # --- execute step ---
        next_state = physics.step(state, dt, **dependent_states)
        return next_state

    def _gather_dependencies(self, dependencies, collectivestate, result_dict):
        for name, deps in dependencies.items():
            dep_states = []
            if isinstance(deps, (tuple,list)):
                for dep in deps:
                    dep_states += list(collectivestate[dep])
            else:
                dep_states = collectivestate[deps]
            result_dict[name] = dep_states
        return result_dict

    def _all_dependencies_fulfilled(self, dependencies, all_states, computed_states):
        state_dict = self._gather_dependencies(dependencies, all_states, {})
        for name, states in state_dict.items():
            for state in states:
                if state.trajectorykey not in computed_states:
                    return False
        return True

    def for_(self, state):
        return self._physics[state.trajectorykey] if state.trajectorykey in self._physics else state.default_physics()

    def add(self, trajectorykey, physics):
        self._physics[trajectorykey] = physics

    def remove(self, trajectorykey):
        del self._physics[trajectorykey]
