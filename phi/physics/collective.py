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

    # def __sub__(self, other):
    #     if isinstance(other, CollectiveState):
    #         return CollectiveState(self._states - other._states)
    #     if isinstance(other, State):
    #         return CollectiveState(self._states - {other})
    #     if isinstance(other, (tuple, list)):
    #         return CollectiveState(self._states - set(other))
    #     raise ValueError("Illegal operation: CollectiveState - %s" % type(other))

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
            else: raise ValueError('State %s not part of CollectiveState' % item)
        if isinstance(item, TrajectoryKey):
            states = list(filter(lambda s: s.trajectorykey==item, self._states))
            assert len(states) == 1, 'CollectiveState[%s] returned %d states' % (item, len(states))
            return states[0]
        if isinstance(item, six.string_types):
            return self.get_by_tag(item)
        if isinstance(item, (tuple, list)):
            return [self[i] for i in item]
        raise ValueError('Illegal argument: %s' % item)

    def default_physics(self):
        phys = CollectivePhysics()
        for state in self._states:
            phys.add(state.trajectorykey, state.default_physics())
        return phys

    def __repr__(self):
        return '[' + ', '.join((str(s) for s in self._states)) + ']'


class CollectivePhysics(Physics):

    def __init__(self):
        Physics.__init__(self, {})
        self._physics = {}  # map from TrajectoryKey to Physics

    def step(self, collectivestate, dt=1.0, **dependent_states):
        assert len(dependent_states) == 0
        next_states = []
        for state in collectivestate.states:
            next_state = self.substep(state, collectivestate, dt)
            next_states.append(next_state)
        return CollectiveState(next_states, age=collectivestate.age + dt)

    def substep(self, state, collectivestate, dt):
        physics = self.for_(state)
        dependent_states = {}
        for name, deps in physics.dependencies.items():
            dep_states = []
            if isinstance(deps, (tuple,list)):
                for dep in deps:
                    dep_states += list(collectivestate[dep])
            else:
                dep_states = collectivestate[deps]
            dependent_states[name] = dep_states
        next_state = physics.step(state, dt, **dependent_states)
        return next_state

    def for_(self, state):
        return self._physics[state.trajectorykey] if state.trajectorykey in self._physics else state.default_physics()

    def add(self, trajectorykey, physics):
        self._physics[trajectorykey] = physics

    def remove(self, trajectorykey):
        del self._physics[trajectorykey]
