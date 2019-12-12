# Simulation architecture

This document discusses the code design of simulations and solvers within Φ<sub>*Flow*</sub>.
If you are interested in how specific simulations work, check out their respective documentations, e.g.
[Fluid](Fluid_Simulation.md).

### States and Physics objects

For each simulation type, there are two distinct entities:

- `State` store a snapshot of a physical system at one point in time.
- `Physics` objects implement the actual solver functionality to progress a simulation forward.

`State` and `Physics` objects can exist independently of each other.
They only interact through the `step` method defined in [`phi.physics.physics.py`](../phi/physics/physics.py).

```python
def step(self, state, dt=1.0, **dependent_states)
```

The `step` method takes one state of a system as input and computes the state of that system at time `t + dt`.
It does not alter the given state in any way, though. Instead it returns a new state.

Once created, a state can never be changed. Every function that changes the state of a system must instead return a copy of the state.

`Physics` objects themselves are stateless, except for algorithmic or hardware-specific details.
All properties of the physical system are stored in the state.

Let's look at an example of a fluid simulation that directly uses `State`s and `Physics`.

```python
from phi.flow import *

inflow = Inflow(box[10:20, 30:34], rate=0.1)
state0 = Fluid(Domain([64, 64]), density=0, velocity=0)
state1 = INCOMPRESSIBLE_FLOW.step(state0, dt=1.0, inflows=[inflow])
```

The first two lines after the import create immutable state objects. The last line executes a simulation step using the global [`Physics`](../phi/physics/physics.py) object INCOMPRESSIBLE_FLOW.

We could also create our own physics object, e.g. to use a specific [pressure solver](Pressure_Solvers.md), or modify the default physics object:

```python
physics = IncompressibleFlow(pressure_solver=custom_solver)
physics.step(state0)
# or
INCOMPRESSIBLE_FLOW.pressure_solver = custom_solver
```

Note that `IncompressibleFlow.step()` takes the additional optional arguments `inflows` and `obstacles`.
These can be used to pass states of different simulations that interact with the fluid.
In `Physics.step`, these arguments are referred to as `**dependent_states`but
each `Physics` implementation can define which ones are appropriate for them.

The following table outlines the most important properties of `States` and `Physics` objects.

|                 | States                                                                                                | Physics                                                                                  |
|-----------------|-------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| Information     | States contain all information about a system at one point in time. The data can change over time.    | Physics objects describe the laws required to evolve one state in time.                  |
| Base class      | [State](../phi/physics.physics.py)(extends [Struct](../phi/struct/stuct.py))                          | [Physics](../phi/physics/physics.py) [(phi.physics.physics)](../phi/physics/physics.py) |
| Mutability      | Immutable                                                                                             | Stateless                                                                                |
| Serialization   | NumPy arrays                                                                                          |                                                                                          |
| Example classes | `StaggeredGrid`, `Fluid`, `Obstacle`, `Inflow`,                                                  | `IncompressibleFlow`, `BurgersPhysics`, `Static`, `GeometryMovement`                            |

## NumPy vs TensorFlow

All subclasses of `Physics` should make sure their code can take both NumPy arrays and TensorFlow tensors as input.
If the input state consists purely of NumPy arrays, `step` will directly compute the next state.
The returned state will also only contain NumPy arrays.

If any property of the input state is a TensorFlow tensor, all computations that depend on this property will be executed using TensorFlow instead. This typically does not execute the computation directly but instead produces a symbolic node.
Consequently, the `step()` method will build a TensorFlow graph and the returned state will contain nodes of that graph.

To actually run the solver with TensorFlow, a `phi.tf.Session` object must be used. The example below shows, how this can be achieved.

```python
from phi.tf.flow import *

session = Session(Scene.create('test'))
inflow = Inflow(box[10:20, 30:34], rate=0.1)
state_in = Fluid(Domain([64, 64]), density=placeholder, velocity=placeholder)
state_out = INCOMPRESSIBLE_FLOW.step(state_in, dt=1.0, inflows=[inflow])
state0 = Fluid(Domain([64, 64]), density=0)
state1 = session.run(state_out, {state_in: state0})
```

This aligns with the typical TensorFlow workflow where `tf.Session.run` is used to execute a graph.
`phi.tf.session.Session.run` simply extends the functionality by allowing `State` objects to be passed directly.

While this extra code is unavoidable for machine learning applications, if you are simply running a simulation, you
can add the states to a world using `world.add(state)` and call the function
`tf_bake_graph(world, session)` to automatically convert all physics objects to TensorFlow graph executions.

The similarities and differences of NumPy vs TensorFlow are illustrated in the example 
[manual_fluid_numpy_or_tf.py](../demos/manual_fluid_numpy_or_tf.py) for a simple custom fluid simulation.

## Simplified API with world

Worlds provide a simplified interface in which states seem mutable (though in reality they are not).
This is achieved in the following way:
The functions `world.X` with the same signature as the constructor `X` create an indirection.
They do not return a state but rather a pointer to a state.
When `world.step` is called or a property of a pointer is changed, a copy of the original states is created.
The old states are then replaced by the new ones and all the pointers redirected to the new states.

Internally, unique names (strings) are used to track the evolution of states.
The returned state of any `Physics.step` call must always have the same `name` property.

Additionally, the world manages dependencies between simulations.
The dependencies are stored in the `Physics` objects as either tags or names.
Tags are unique identifier strings that allow simulations to find other simulations.
The world (or more precisely the [CollectivePhysics](../phi/physics/collective.py)) then finds all states that match the dependencies and passes them to the `step` method.
