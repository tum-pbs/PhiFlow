# Simulations in Φ<sub>*Flow*</sub>

This document gives an overview of how to run simulations using Φ<sub>*Flow*</sub>.
For a deeper look into how the code is structured, check out [the simulation architecture documentaiton](Simulation_Architecture.md).
If you are interested in how specific simulations work, check out their respective documentations, e.g.
[Fluid](Fluid_Simulation.md).
A paragraph below also summarizes differences to mantaflow simulations.

## Running simulations

### Simulation references and worlds

The easiest way to run a simulation or set of interacting simulations forward in time is using the `World` interface.
A `World` manages the current states of registered simulations as well as the physical behaviour that drives them.

The following code uses the default world to initialize a fluid simulation and create an inflow.

```python
from phi.flow import *

fluid = world.Fluid(Domain([80, 64]), density=0)
world.Inflow(Sphere((10, 32), 5), rate=0.2)
```

Most properties of the fluid simulation can be accessed and manipulated through the simulation reference, `fluid` in this case.
This includes fields (e.g. `fluid.density`, `fluid.velocity`) as well as properties (e.g. `fluid.buoyancy_factor`).

Note that while `fluid.density` directly returns a NumPy array, the velocity is held in staggered form and
`fluid.velocity` returns an instance of `StaggeredGrid`.
To access the actual NumPy array holding the staggered values, write `fluid.velocity.staggered` (read-only).
For more on staggered grids, see [the documentation](Staggered_Grids.md).

To run a simulation, we can use the `step` method:

```python
world.step(dt=1.0)
```

This progresses all simulations that are associated with that world by a time increment `dt` (defaults to `1.0`).
Accessing any property of a simulation reference (such as `fluid`) will now return the updated value.

### Simulation + GUI

To use the browser-based GUI that comes with Φ<sub>*Flow*</sub>, we need to wrap our simulation code with a
`App`.
How to use `App`s is documented [here](Web_Interface.md).

The following code is taken from [the simpleplume example](../demos/simpleplume.py).

```python
class Simpleplume(App):

    def __init__(self):
        App.__init__(self, 'Simpleplume', stride=5)
        fluid = world.Fluid(Domain([80, 64], CLOSED))
        world.Inflow(Sphere((10, 32), 5), rate=0.2)
        self.add_field('Density', lambda: fluid.density)
        self.add_field('Velocity', lambda: fluid.velocity)

show()
```

By subclassing `App`, we inherit the following functions:

- `add_field` to make data visible in the GUI
- `show` which launches the web service
- `step` which calls `world.step()` by default but can be overwritten to implement custom behaviour

Slightly more complex examples can be found in 
[marker.py](../demos/marker.py) which passively advects an additional marker field,
[fluid_logo.py](../demos/fluid_logo.py) which adds obstacles to the scene and
[moving_inflow.py](../demos/moving_inflow.py) which adds geometry movement.
The example [manual_fluid_numpy_or_tf.py](../demos/manual_fluid_numpy_or_tf.py) shows a simple
fluid simulation with custom inflow, that can run via NumPy or TensorFlow with a switch,
to illustrate similarities and differences of the two execution modes.

### Running on the GPU

For GPU execution, TensorFlow needs to be installed (see the [installation instructions](Installation_Instructions.md)).
To run the simulation using TensorFlow, change the first line `from phi.flow import *` to `from phi.tf.flow import *`.
This replaces the imported `App` with a TensorFlow-enabled version.
This new `App` bakes the physics into a TensorFlow graph in `show()` before the GUI is launched.

## Differences to MantaFlow

[MantaFlow](http://mantaflow.com/) is a simulation framework that also offers
deep learning functionality and TensorFlow integration. However, in contrast to
Φ<sub>*Flow*</sub>, it focuses on fast CPU-based simulations, and does not
support differentiable operators. Nonetheless, it can be useful to, e.g.,
pre-compute simulation data for learning tasks in Φ<sub>*Flow*</sub>.

One central difference of both fluid solvers is that mantaflow grids all have
the same size, while in Φ<sub>*Flow*</sub>, the staggered velocity grids are
larger by one layer on the positive domain sides 
(also see the [data format section](Scene_Format_Specification.md)).

Mantaflow always stores 3-component vectors in its `Vec3`
struct, while Φ<sub>*Flow*</sub> changes the vectors size with the
dimensionality of the solver. E.g., a 2D solver in mantaflow with a velocity `Vec3 v`
has `v[0]` for X, and `v[1]` for the Y component. `v[2]` for Z is still
defined, but typically set to zero. For a 3D solver in mantaflow, this indexing
scheme does not change.

Φ<sub>*Flow*</sub>, on the other hand, uses a two component numpy array for the
velocity component of a 2D solver, where the last index always denotes X. I.e.,
for a vector `w` the expression `w[-1]` can be used to access X, but
as `w` has only two components, `w[0]` denotes Y, and `w[1]` X in this case.
Correspondingly,
`w[0]`, `w[1]` and `w[2]` denote Z,Y and X for a 3D run. This scheme is
closer to the typical ordering of numpy arrays, and simplifies the
implementation of operators that are agnostic of the solver dimension.

As a side effect of this indexing, gravity is defined to act along the first
dimension in Φ<sub>*Flow*</sub>. I.e., Z in 3D, and Y in 2D. In contrast,
mantaflow's gravity always acts along the Y direction.
