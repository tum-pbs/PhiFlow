from phi.math import Struct, zeros


class State(Struct):

    def __init__(self, tags=()):
        self._tags = tuple(tags)

    @property
    def tags(self):
        return self._tags

    def __mul__(self, operation):
        return operation(self)


class Physics(object):

    def __init__(self):
        """
A Physics object describes a set of physical laws that can be used to simulate a system by moving from state to state,
tracing out a trajectory.
The description of the physical systems (e.g. obstacles, boundary conditions) is also included in the Physics object
and the enclosing world.
        :param world: (optional) the world this system lives in, used to update the worldstate when change
        :param dt: simulation time increment
        :param identifier: This Physics acts on all states with this tag
        """
        self.dt = 1.0
        self._worldstate = None

    @property
    def worldstate(self):
        if self._worldstate is not None:
            return self._worldstate
        else:
            from .world import world
            return world.state

    @worldstate.setter
    def worldstate(self, value):
        self._worldstate = value

    def step(self, state):
        """
Computes the next state of the simulation, given the current state.
Solves the simulation for a time increment self.dt.
        :param state: current state
        :return next state
        """
        raise NotImplementedError(self)

    def shape(self, batch_size=1):
        raise NotImplementedError(self)

    def serialize_to_dict(self):
        return {'type': self.__class__.__name__}

    def unserialize_from_dict(self):
        pass

    def initial_state(self, batch_size=1):
        return zeros(self.shape())
