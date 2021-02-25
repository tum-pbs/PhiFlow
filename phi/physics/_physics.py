"""
Defines base classes for all simulations such as State and Physics.
"""

from phi import struct


@struct.definition()
class State(struct.Struct):
    """
    States describe one configuration of a physical system.
    
    State objects are generally immutable, i.e. once created the state cannot be changed.
    Instead, new State objects are created when the state changes.
    This guarantees that previously acquired states are not altered by other parts of the code which is convenient when working with sequence values.
    
    States are identified by their unique name.

    Args:

    Returns:

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

        Args:
          tags: 

        Returns:

        """
        return tuple(tags)

    @struct.variable(default=0.0, holds_data=False)
    def age(self, age):
        """
        Cumulative dt of all step() invocations. States usually start out at age=0.

        Args:
          age: 

        Returns:

        """
        return age

    @struct.constant()
    def name(self, name):
        """
        Names uniquely identify the system represented by this state.
        All states that represent a configuration of the same system must have the same name.
        
        Names can also be used as a shortcut to reference states (e.g. in StateCollection or World).

        Args:
          name: 

        Returns:

        """
        if name is None:
            return '%s_%d' % (self.__class__.__name__.lower(), id(self))
        else:
            return str(name)

    def default_physics(self):
        """Returns a Physics object that can be used to progress this state forward in time."""
        return STATIC

    @property
    def state(self):
        """:return: self"""
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
        blocking_str = 'updated' if self.blocking else 'previous'
        single_str = 'single' if self.single_state else 'all'
        if self.state_name is not None:
            return "'%s' referencing %s %s '%s'" % (self.parameter_name, single_str, blocking_str, self.state_name)
        else:
            return "'%s' containing %s %s %s" % (self.parameter_name, single_str, blocking_str, self.tag)


class Physics(object):
    """
    A Physics object describes a set of physical laws that can be used to simulate a system by moving from state to state, tracing out a trajectory.
    Physics objects are stateless and always support an empty constructor.

    Args:

    Returns:

    """

    def __init__(self, dependencies=()):
        self.dependencies = tuple(dependencies)

    def step(self, state, dt=1.0, **dependent_states):
        """
        Computes the next state of a physical system, given the current state.
        Solves the simulation for a time increment self.dt.

        Args:
          state: current state
          dt: time increment, float (can be positive, negative or zero) (Default value = 1.0)
          dependent_states: dict from String to List<State>
          **dependent_states: 

        Returns:
          next state of the same type as state

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

    @property
    def dependency_names(self):
        return [dep.parameter_name for dep in self.dependencies]


class Static(Physics):
    """Physics for states with no natural evolution."""

    def step(self, state, dt=1.0, **dependent_states):
        """
        Does not alter the state except for increasing its age.

        Args:
          state: 
          dt:  (Default value = 1.0)
          **dependent_states: 

        Returns:

        """
        return state.map_item(State.age, lambda age: age + dt)


STATIC = Static()


class _ChainedPhysics(Physics):

    def __init__(self, physics_list):
        physics_list = [_as_physics(physics) for physics in physics_list]
        Physics.__init__(self, dependencies=sum([phys.dependencies for phys in physics_list], ()))
        self.physics_list = physics_list
        assert len(set(self.dependency_names)) == len(self.dependencies), 'Duplicate dependency parameter names: %s' % self.dependencies

    def step(self, state, dt=1.0, **dependent_states):
        for physics in self.physics_list:
            deps = {key: value for key, value in dependent_states.items() if key in physics.dependency_names}
            state = physics.step(state, dt=dt, **deps)
        return state.map_item(State.age, lambda age: age + dt)


class _WrapPhysics(Physics):

    def __init__(self, function):
        Physics.__init__(self, dependencies=())
        self.function = function

    def step(self, state, dt=1.0, **dependent_states):
        result = self.function(state, dt)
        return result


def _as_physics(physics_like):
    if isinstance(physics_like, Physics):
        return physics_like
    if isinstance(physics_like, (tuple, list)):
        return _ChainedPhysics(physics_like)
    if callable(physics_like):
        return _WrapPhysics(physics_like)
    raise ValueError("Cannot convert '%s' to Physics" % physics_like)
