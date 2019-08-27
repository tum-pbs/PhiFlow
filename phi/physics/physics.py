from phi.math import Struct, struct, staticshape


class TrajectoryKey(object):
    """
    Used to identify State objects that are part of one trajectory.
    States from the same trajectory reference the same TrajectoryKey object.
    TrajectoryKey objects use their object identity with the default equals and hash implementation.
    """
    pass


class State(Struct):

    __struct__ = struct.Def((), ('_age', '_tags'))

    def __init__(self, tags=(), age=0.0, batch_size=None):
        self._tags = tuple(tags)
        self.trajectorykey = TrajectoryKey()
        self._age = age
        self._batch_size = batch_size

    @property
    def tags(self):
        return self._tags

    @property
    def age(self):
        return self._age

    def default_physics(self):
        return STATIC

    @property
    def shape(self):
        def tensorshape(tensor):
            if tensor is None: return None
            default_batched_shape = staticshape(tensor)
            if len(default_batched_shape) >= 2:
                return (self._batch_size,) + default_batched_shape[1:]
        return struct.map(tensorshape, self)


class Physics(object):
    """
    A Physics object describes a set of physical laws that can be used to simulate a system by moving from state to state,
tracing out a trajectory.
    Physics objects are stateless and always support an empty constructor.
    """

    def __init__(self, dependencies=None, blocking_dependencies=None):
        self.dependencies = dependencies if dependencies is not None else {}  # Map from String to List<tag or TrajectoryKey>
        self.blocking_dependencies = blocking_dependencies if blocking_dependencies is not None else {}

    def step(self, state, dt=1.0, **dependent_states):
        """
Computes the next state of the simulation, given the current state.
Solves the simulation for a time increment self.dt.
        :param state: current state
        :param dependent_states: dict from String to List<State>
        :param dt: time increment (can be positive, negative or zero)
        :return next state
        """
        raise NotImplementedError(self)


class Static(Physics):

    def step(self, state, dt=1.0, **dependent_states):
        return state.copied_with(age=state.age + dt)


STATIC = Static()