from .physics import *


class CollectiveState(State):

    def __init__(self, states=(), time=0.0):
        State.__init__(self, ('collection',))
        self._states = states if isinstance(states, set) else set(states)
        self.time = time

    def __add__(self, other):
        if isinstance(other, CollectiveState):
            return CollectiveState(self._states | other._states)
        if isinstance(other, State):
            return CollectiveState(self._states | {other})
        if isinstance(other, (tuple, list)):
            return CollectiveState(self._states | set(other))
        raise ValueError("Illegal operation: CollectiveState + %s" % type(other))

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, CollectiveState):
            return CollectiveState(self._states - other._states)
        if isinstance(other, State):
            return CollectiveState(self._states - {other})
        if isinstance(other, (tuple, list)):
            return CollectiveState(self._states - set(other))
        raise ValueError("Illegal operation: CollectiveState - %s" % type(other))

    def get_by_tag(self, tag):
        return [o for o in self._states if tag in o.tags]

    @property
    def states(self):
        return self._states

    def disassemble(self):
        tensors, reassemble_states = list_tensors(self._states)
        def reassemble(tensors):
            return CollectiveState(reassemble_states(tensors), self.time)
        return tensors, reassemble

    def __getitem__(self, item):
        if isinstance(item, Physics):
            states = self.get_by_tag(item.state_tag)
            if len(states) != 1:
                raise ValueError("[%s] returned %d entries, but required 1" % (item, len(states)))
            return states[0]
        raise ValueError("Illegal index: %s" % item)


class CollectivePhysics(Physics):

    def __init__(self, world, dt=1.0):
        Physics.__init__(self, world, state_tag='', dt=dt)
        self._physics = []

    def step(self, collectivestate):
        current_states = list(collectivestate.states)
        next_states = []

        for physics in self._physics:
            physics.dt = self.dt
            for state in collectivestate.get_by_tag(physics.state_tag):
                next_states.append(physics.step(state))
                current_states.remove(state)

        # Static states
        next_states += current_states

        return CollectiveState(next_states, collectivestate.time + self.dt)

    def shape(self, batch_size=1):
        shapes = [phys.shape(batch_size) for phys in self._physics]
        return CollectiveState(shapes)

    def add(self, physics):
        self._physics.append(physics)

    def remove(self, physics):
        self._physics.remove(physics)

    def serialize_to_dict(self):
        return [phys.serialize_to_dict() for phys in self._physics]

    @property
    def all(self):
        return self._physics
