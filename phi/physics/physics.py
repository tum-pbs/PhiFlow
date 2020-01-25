"""
Defines base classes for all simulations such as State and Physics.
"""

from phi import struct
from phi.math import staticshape


@struct.definition()
class State(struct.Struct):
    """
    States describe one configuration of a physical system.

    State objects are generally immutable, i.e. once created the state cannot be changed.
    Instead, new State objects are created when the state changes.
    This guarantees that previously acquired states are not altered by other parts of the code which is convenient when working with sequence data.

    States are identified by their unique name.
    """

    def __init__(self, batch_size=None, **kwargs):
        self._batch_size = batch_size
        struct.Struct.__init__(self, **kwargs)

    @struct.constant(default=())
    def tags(self, tags):
        """
Tags are used to resolve dependencies.
They represent traits or classes of the state.

Physics objects typically definition their dependencies in terms of tags.
        """
        return tuple(tags)

    @struct.variable(default=0.0, holds_data=False)
    def age(self, age):
        """
Cumulative dt of all step() invocations. States usually start out at age=0.
        """
        return age

    @struct.constant()
    def name(self, name):
        """
Names uniquely identify the system represented by this state.
All states that represent a configuration of the same system must have the same name.

Names can also be used as a shortcut to reference states (e.g. in StateCollection or World).
        """
        if name is None:
            return '%s_%d' % (self.__class__.__name__.lower(), id(self))
        else:
            return str(name)

    def default_physics(self):
        """
Returns a Physics object that can be used to progress this state forward in time.
        """
        return STATIC

    @property
    def shape(self):
        """
Similar to phi.math.shape(self) but respects unknown dimensions.
        """
        def tensorshape(tensor):
            if tensor is None:
                return None
            default_batched_shape = staticshape(tensor)
            if len(default_batched_shape) >= 2:
                return (self._batch_size,) + default_batched_shape[1:]
            else:
                return default_batched_shape
        with struct.unsafe():
            return struct.map(tensorshape, self, item_condition=struct.VARIABLES)

    @property
    def state(self):
        """
        :return: self
        """
        return self

    def __repr__(self):
        return '%s[name="%s"]' % (self.__class__.__name__, self.name)


class StateDependency(object):
    # pylint: disable-msg = too-few-public-methods
    """
StateDependencies can be used by Physics objects to request information about other states.
    """

    def __init__(self, parameter_name, tag, single_state=False, blocking=False, state_name=None):
        """
Define a StateDependency.
        :param parameter_name: the state(s) will be passed to step() under this name
        :param tag: Model dependency by tag
        :param single_state: If True: checks that only one state matches the criteria. This state is then passed. If False: passes a list of matching states.
        :param blocking: If True: computes all states matching the criteria before calling step() on this Physics. This Physics will be passed the updated versions of dependent states. If False: step() will be passed the previous versions of dependent states.
        :param state_name: If not None, references a specific physical system by name instead of using a tag. In this case single_state must be set to True.
        """
        self.parameter_name = parameter_name
        self.tag = tag
        self.single_state = single_state
        self.blocking = blocking
        self.state_name = state_name
        if state_name is not None:
            assert single_state

    def __repr__(self):
        if self.state_name is not None:
            return '[key=%s, blocking=%s]' % (self.state_name, self.blocking)
        else:
            return '[tag=%s, blocking=%s]' % (self.tag, self.blocking)


class Physics(object):
    """
    A Physics object describes a set of physical laws that can be used to simulate a system by moving from state to state, tracing out a trajectory.
    Physics objects are stateless and always support an empty constructor.
    """

    def __init__(self, dependencies=()):
        self.dependencies = tuple(dependencies)

    def step(self, state, dt=1.0, **dependent_states):
        """
        Computes the next state of a physical system, given the current state.
        Solves the simulation for a time increment self.dt.

        :param state: current state
        :param dt: time increment, float (can be positive, negative or zero)
        :param dependent_states: dict from String to List<State>
        :return next state of the same type as state
        """
        raise NotImplementedError(self)

    @property
    def blocking_dependencies(self):
        # pylint: disable-msg = missing-function-docstring
        return filter(lambda d: d.blocking, self.dependencies)

    @property
    def non_blocking_dependencies(self):
        # pylint: disable-msg = missing-function-docstring
        return filter(lambda d: not d.blocking, self.dependencies)


class Static(Physics):
    """
Physics for states with no natural evolution.
    """

    def step(self, state, dt=1.0, **dependent_states):
        """
Does not alter the state except for increasing its age.
        """
        return state.copied_with(age=state.age + dt)


STATIC = Static()
