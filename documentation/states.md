# States

Introduced in Φ-*Flow* v0.3, simulations are split into a temporally evolving state and a fixed physics object that encodes the time evolution.


|                 | States                                                                                             | Physics                                                                                                |
|-----------------|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Information     | States contain all information about a system at one point in time. The data can change over time. | Physics contain all laws required to evolve one state in time. The properties do not evolve over time. |
| Base class      | State(Struct)                                                                                      | Physics                                                                                                |
| Mutability      | Immutable                                                                                          | Mutable                                                                                                |
| Serialization   | NumPy arrays                                                                                       | dict / JSON                                                                                            |
| Example classes | ndarray, Tensor, StaggeredGrid, SmokeState, Obstacle, Inflow                                       | Smoke, Burger, StaticObject, DynamicObject                                                             |t                                                     |

The following code snippet sets up a Physics object and a corresponding state, then computes the next state.
```python
from phi.flow import *
physics = Smoke(Domain([80, 64], SLIPPERY))
state = physics.initial_state()
next_state = physics.step(state)
```

