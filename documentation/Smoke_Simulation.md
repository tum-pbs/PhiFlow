# Smoke simulation

A Eulerian smoke simulation typically tracks a smoke density and air velocity field.
The physical behaviour can be computed using the following steps

- The smoke creates a buoyancy force, proportional to the smoke density.
- The velocity field is made divergence-free to enforce incompressibility.
- The smoke moves with the air, i.e. the density is advected using the velocity.
- The air itself moves, i.e. the velocity is self-advected.

In Φ<sub>*Flow*</sub>, a smoke simulation can be initiated by creating a smoke state.
```python
from phi.flow import *
smoke = Smoke(Domain([64, 64]), density=0, velocity=0, buoyancy_factor=0.1, conserve_density=True)
```

The `step` method, defined in [`SmokePhysics`](../phi/physics/smoke.py), executes the steps mentioned above.

The velocity of a smoke state is sampled in [staggered form](Staggered_Grids.md), i.e. an instance of
[`StaggeredGrid`](../phi/math/nd.py).

For the pressure solve, a [`PressuresSolver`](../phi/solver/base.py) object is managed by the `SmokePhysics`.
The [documentation on pressure solvers](Pressure_Solvers.md) explains the differences between the available solvers.
The example [runsim_numpy_or_tf.py](../apps/runsim_numpy_or_tf.py) shows how the 
sequence of simulation steps of a smoke simulation can be executed manually without
using the `SmokePhysics` object.

