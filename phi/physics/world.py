from .collective import CollectiveState, CollectivePhysics
from .smoke import *
from .burger import *
from .heat import *
from .obstacle import *
from .effect import *
import inspect


class StateProxy(object):
    """
    StateProxy mirrors all attributes of a state in an associated world.
    While State objects are generally immutable, StateProxy also implements setting any attribute of the state.
    When an attribute is set, a copy of the state with the new value replaces the old state in the world.
    This object then mirrors the values of the new state.

    After world.step() is invoked, all Proxies of that world will mirror the state after stepping.

    To reference the current immutable state of
    """

    def __init__(self, world, trajectorykey):
        self.world = world
        self.trajectorykey = trajectorykey

    @property
    def state(self):
        state = self.world.state[self.trajectorykey]
        assert state is not None
        return state

    @state.setter
    def state(self, state):
        self.world.state = self.world.state.state_replaced(self.state, state)

    @property
    def physics(self):
        physics = self.world.physics.for_(self.state)
        assert physics is not None
        return physics

    @physics.setter
    def physics(self, phys):
        self.world.physics.add(self.trajectorykey, phys)

    def step(self, dt=1.0, physics=None):
        self.world.step(self, dt=dt, physics=physics)

    def __getattr__(self, item):
        assert item not in ('world', 'trajectorykey', 'physics', 'state')
        return getattr(self.state, item)

    def __setattr__(self, key, value):
        if key in ('world', 'trajectorykey', 'physics', 'state'):
            object.__setattr__(self, key, value)
        else:
            self.state = self.state.copied_with(**{key:value})





def _proxy_wrap(world, constructor):
    try:
        const_args = inspect.getargspec(constructor)[0]
        static = const_args[0] != 'self'
    except:
        static = False
        const_args = ['batch_size']
    def buildadd(*args, **kwargs):
        if 'batch_size' not in kwargs and 'batch_size' in const_args:
            kwargs['batch_size'] = world.batch_size
        if 'physics' in kwargs:
            physics = kwargs['physics']
            assert physics is not None
            del kwargs['physics']
        else: physics = None
        invok = constructor.__func__ if static else constructor
        state = invok(*args, **kwargs)
        world.add(state)
        proxy = StateProxy(world, state.trajectorykey)
        if physics is not None:
            proxy.physics = physics
        return proxy
    return buildadd


class World(object):
    """
    A world object defines a global state as well as a set of rules (Physics objects) that define how the state evolves.

    The world manages dependencies among the contained simulations and provides convenience methods for creating proxies for specific simulations.

    The method world.step() evolves the whole state or optionally a specific state in time.
    """

    def __init__(self):
        self._state = CollectiveState()
        self.physics = self._state.default_physics()
        self.observers = set()
        self.batch_size = None
        # --- Insert object / create proxy shortcuts ---
        for proxy in ('Smoke', 'Burger', 'Obstacle', 'Inflow', 'Fan', 'ConstantDensity',
                      'Heat', 'ConstantTemperature', 'HeatSource', 'ColdSource'):
            setattr(self, proxy, _proxy_wrap(self, getattr(self, proxy)))

    Smoke = Smoke
    Burger = Burger
    Obstacle = Obstacle
    Inflow = Inflow
    Fan = Fan
    ConstantDensity = ConstantDensity
    Heat = Heat
    ConstantTemperature = ConstantTemperature
    HeatSource = HeatSource
    ColdSource = ColdSource


    @property
    def state(self):
        return self._state

    @property
    def age(self):
        return self._state.age

    @state.setter
    def state(self, state):
        assert state is not None
        self._state = state
        for observer in self.observers: observer(self)

    def step(self, state=None, dt=1.0, physics=None):
        """
Evolves the current world state by a time increment dt.
If state is provided, only that state is evolved, leaving the others untouched.
The optional physics parameter can then be used to override the default physics.
Otherwise, all states are evolved.

Calling World.step resolves all dependencies among simulations and then calls Physics.step on each simulation to evolve the states.

Invoking this method alters the world state. To to_field a copy of the state, use :func:`World.stepped <~world.World.stepped>` instead.
        :param state: State, StateProxy or None
        :param dt: time increment, default 1.0
        :param physics: Physics object for the state or None for default
        :return: evolved state if a specific state was provided
        """
        if state is None:
            if physics is None: physics = self.physics
            self.state = physics.step(self._state, dt)
        else:
            if isinstance(state, StateProxy):
                state = state.state
            s = self.physics.substep(state, self._state, dt, override_physics=physics)
            self.state = self._state.state_replaced(state, s).copied_with(age=self._state.age + dt)
            return s

    def stepped(self, state=None, dt=1.0, physics=None):
        if state is None:
            if physics is None: physics = self.physics
            return physics.step(self._state, None, dt)
        else:
            if isinstance(state, StateProxy):
                state = state.state
            return self.physics.substep(state, self._state, dt, override_physics=physics)

#     def on_change(self, observer):
#         """
# Register an observer that will be called when states are added to the world or removed from the world.
# The observer must define __call__ and will be given the world as parameter.
#         :param observer:
#         """
#         self.observers.add(observer)
#
#     def remove_on_change(self, observer):
#         self.observers.remove(observer)

    def add(self, state, physics=None):
        self.state += state
        if physics is not None:
            self.physics.add(state.trajectorykey, physics)

    def remove(self, state):
        self.state -= state
        self.physics.remove(state.trajectorykey)

    def clear(self):
        self._state = CollectiveState()
        self.physics = self._state.default_physics()
        self.state = self._state


world = World()
