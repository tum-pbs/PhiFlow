# Fluid simulation

The default fluid simulation in Φ<sub>Flow</sub> describes a single-phase flow such as air or a filled tank of water.
It tracks the fluid's velocity field as well as a marker density which can optionally be used to produce a buoyancy force.
The physical behaviour can be computed using the following steps

- **Advection**: The marker density and velocity field are transported in the direction given by the velocity field.
- The marker may create a **buoyancy** force, proportional to its density.
- The velocity field is made divergence-free to enforce **incompressibility**.


## `Fluid` and `IncompressibleFlow`

In Φ<sub>Flow</sub>, a fluid simulation can be initiated by creating a `Fluid` state and an `IncompressibleFlow` physics object.

```python
from phi.flow import *

initial_state = Fluid(Domain([64, 64]), density=0, velocity=0, buoyancy_factor=0.1)
world.add(initial_state, physics=IncompressibleFlow())

```

The `step` method, defined in [`IncompressibleFlow`](../phi/physics/fluid.py), executes the steps mentioned above.
When added to a `world`, it is called by `world.step()`.

The velocity of a fluid state is sampled in [staggered form](Staggered_Grids.md), i.e. an instance of
[`StaggeredGrid`](../phi/physics/field/staggered_grid.py) while the density is a [`CenteredGrid`](../phi/physics/field/grid.py).
For more on [Fields, see here](Fields.md).

For the pressure solve, a [`PressuresSolver`](../phi/physics/pressuresolver/base.py) object is managed by the `IncompressibleFlow`.
The [documentation on pressure solvers](Pressure_Solvers.md) explains the differences between the available solvers.
The example [manual_fluid_numpy_or_tf.py](../demos/manual_fluid_numpy_or_tf.py) shows how the
sequence of simulation steps of a fluid simulation can be executed manually without
using the `IncompressibleFlow` object.


## Algorithmic Details

The Navier-Stokes fluid solver in phiflow by default uses a ‘’stable fluids’’-type fluid solver [Stam 1999].
More specifically it employs:
- a [staggered  velocity grid](Staggered_Grids.md) (MAC) [Harlow and Welch 1965] for central finite-differences,
- implicit Chorin pressure projection [Chorin and Temam 1968] with a CG solver,
- first-order Semi-Lagrangian advection [Stam 1999],
- Boussinesq approximation for buoyancy [Boussinesq 1897],
- no viscosity solve, i.e., only numerical viscosity is in effect,
- with semi-explicit time-stepping with operator splitting (first order).
