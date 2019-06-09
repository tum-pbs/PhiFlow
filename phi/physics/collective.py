from .physics import *
from phi.math import disassemble


class CollectiveState(State):

    def __init__(self, states=(), time=0.0):
        State.__init__(self, ('collection',))
        self._states = states if isinstance(states, tuple) else tuple(states)
        self.time = time

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

    def disassemble(self):
        tensors, reassemble_states = disassemble(self._states)
        def reassemble(tensors):
            return CollectiveState(reassemble_states(tensors), self.time)
        return tensors, reassemble


class CollectivePhysics(Physics):

    def __init__(self):
        Physics.__init__(self)
        self._physics = []

    def step(self, collectivestate):
        assert len(self._physics) == len(collectivestate.states)

        next_states = []
        for physics, state in zip(self._physics, collectivestate.states):
            physics.dt = self.dt
            physics.worldstate = collectivestate
            next_states.append(physics.step(state))

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
