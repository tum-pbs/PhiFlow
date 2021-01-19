# Physics

The module `phi.physics` \[[source](../phi/physics)\] provides a library of common operations used to solve partial differential equations like [fluids](Fluid_Simulation.md).
It builds on the [field](./Fields.md), [geometry](./Geometry.md) and [math](./Math.md) modules and constitutes the highest-level API for physical simulations in Φ<sub>Flow</sub>.
Similar to the field module, physics functions act on data structures represented by the `Field` class.

## Domain and Material
The `physics` module provides the classes `Domain` and `Material` \[[source](../phi/physics/_boundaries.py)\] which simplify defining grids and boundary conditions.

A `Domain` object defines physical properties of the space the simulation lives in and is required for some physics functions.
Specifically, it defines:

* `resolution: Shape`: grid resolution and axis order
* `boundaries: Material`: boundary conditions for scalar fields, vector fields and accessibility fields.
* `bounds: Box` (optional): domain size in a chosen length unit, e.g. meters or millimeters. If not specified, uses cells of unit size.

When creating a domain, the resolution can alternatively be passed keyword arguments listing the dimensions, e.g. `Domain(x=64, y=48)`.

As described in the [fields documentation](./Fields.md), the boundary conditions of fields are implemented as `Extrapolation` objects.
The `Material` class simply stores the extrapolations for various types of fields and makes them easier to work with.

There are a number of predefined materials, most notably `OPEN` and `CLOSED`.
To define domains with varying boundary conditions, simply pass a list of materials or material pairs matching the axis order.

Example:
`Domain(x=64, y=64, boundaries=[CLOSED, (CLOSED, OPEN)]` creates a box with an open top.

Domains can be used to create grids.
The main methods are `grid` and `vector_grid`, each with the arguments `(value, type, extrapolation)`.

* For the `value`, a constant, another `Field` or a function may be passed.
* The `type` must be either `CenteredGrid` (default) or `StaggeredGrid`.
* The `extrapolation` is optional and defaults to `boundaries.grid_extrapolation` for `grid` and `boundaries.vector_extrapolation` for `vector_grid`.

Additionally, `Domain` provides the following convenience methods and shortcuts:
* `staggered_grid()` calls `vector_grid` with `type=StaggeredGrid`.
* `vgrid = vector_grid`
* `sgrid = staggered_grid`

Example:
```python
from phi.flow import *

DOMAIN = Domain(x=64, y=80, boundaries=CLOSED, bounds=Box[0:100, 0:100])
velocity = DOMAIN.staggered_grid(Noise())
pressure = DOMAIN.grid(0)
```


## Writing a Custom Simulation
In previous version of Φ<sub>Flow</sub>, all custom simulations were based on the abstract classes `State` and `Physics`.
This is no longer the case.

It is recommended to define custom re-usable operations as functions and call them directly from the top-level Python script.
The following script runs 100 steps of an inviscid fluid simulation.
```python
from phi.flow import *

DOMAIN = Domain(x=64, y=80, boundaries=CLOSED, bounds=Box[0:100, 0:100])
velocity = DOMAIN.staggered_grid(Noise())
pressure = DOMAIN.grid(0)
for _ in range(100):
    velocity = advect.semi_lagrangian(velocity, velocity, dt=1)
    velocity, pressure, iterations, _ = fluid.make_incompressible(velocity, DOMAIN, pressure_guess=pressure)
```
Note that `advect.semi_lagrangian` and `fluid.make_incompressible` are standard functions in Φ<sub>Flow</sub>, contained in the standard import.


## Visualizing the simulation

The simplest way to visualize a simple simulation with module-level variables (`velocity` and `pressure` in the example above) is to use the `ModuleViewer` class.
In the above example, simply replace the line containing the `for` loop with the following line.
```python
for _ in ModuleViewer().range(100):
```
This launches a web interface displaying the velocity and pressure fields and allows you to step through the simulation step by step.

Slightly more complex examples can be found in 
[marker.py](../demos/marker.py) which passively advects an additional marker field,
[simpleplume.py](../demos/simpleplume.py) which additionally introduces a buoyancy force,
[fluid_logo.py](../demos/fluid_logo.py) which adds obstacles to the scene and
[rotating_bar.py](../demos/rotating_bar.py) which adds geometry movement.


## Running on the GPU

See the [GPU execution guide](GPU_Execution.md).

## Differences to MantaFlow

[MantaFlow](http://mantaflow.com/) is a simulation framework that also offers
deep learning functionality and TensorFlow integration. However, in contrast to
Φ<sub>Flow</sub>, it focuses on fast CPU-based simulations, and does not
support differentiable operators. Nonetheless, it can be useful to, e.g.,
pre-compute simulation data for learning tasks in Φ<sub>Flow</sub>.

One central difference of both fluid solvers is that mantaflow grids all have
the same size, while in Φ<sub>Flow</sub>, the staggered velocity grids are
larger by one layer on the positive domain sides 
(also see the [data format section](Scene_Format_Specification.md)).

Mantaflow always stores 3-component vectors in its `Vec3`
struct, while Φ<sub>Flow</sub> changes the vectors size with the
dimensionality of the solver. E.g., a 2D solver in mantaflow with a velocity `Vec3 v`
has `v[0]` for X, and `v[1]` for the Y component. `v[2]` for Z is still
defined, but typically set to zero. For a 3D solver in mantaflow, this indexing
scheme does not change.

Φ<sub>Flow</sub>, on the other hand, uses a two component numpy array for the
velocity component of a 2D solver, where the first or last index denotes X, depending on the configuration.
This scheme is closer to the typical ordering of numpy arrays, and simplifies the
implementation of operators that are agnostic of the solver dimension.

As a side effect of this indexing, gravity is defined to act along the last or first
dimension in Φ<sub>Flow</sub>, i.e., Z in 3D, and Y in 2D.
In contrast, gravity always acts along the Y direction in Mantaflow.
