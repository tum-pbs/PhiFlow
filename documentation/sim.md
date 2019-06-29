
# Simulations in Φ<sub>*Flow*</sub>

This document gives an overview of how to run simulations using Φ<sub>*Flow*</sub>.
For a deeper look into how the code is structured, check out [the simulation architecture documentaiton](simdesign.md).
If you are interested in how specific simulations work, check out their respective documentations, e.g.
[Smoke](documentation/smoke.md).


## Running simulations

### Simulation references and worlds

The easiest way to run a simulation or set of interacting simulations forward in time, is using the `World` interface.
A `World` manages the current states of registered simulations as well as the physical behaviour that drives them.

The following code uses the default world to initialize a smoke simulation and create an inflow.

```python
from phi.flow import *
smoke = world.Smoke(Domain([80, 64]), density=0)
world.Inflow(Sphere((10, 32), 5), rate=0.2)
```

Most properties of the smoke simulation can be accessed and manipulated through the simulation reference,
`smoke` in this case.
This includes fields such as `smoke.density`, `smoke.velocity` as well as properties like `smoke.buoyancy_factor`.

Note that while `smoke.density` directly returns a NumPy array, the velocity is held in staggered form and
`smoke.velocity` returns an instance of `StaggeredGrid`.
To access the actual NumPy array holding the staggered values, write `smoke.velocity.staggered` (read-only).
For more on staggered grids, see [the documentation](./staggered.md).

To run a simulation, we can use the `step` method:

```python
world.step(dt=1.0)
```

This progresses all simulations that are associated with that world by a time increment `dt` (defaults to `1.0`).
Accessing any property of a simulation reference (such as `smoke`) will now return the updated value.


### Simulation + GUI

To use the browser-based GUI that comes with Φ<sub>*Flow*</sub>, we need to wrap our simulation code with a
`FieldSequenceModel`.
How to use `FieldSequenceModel`s is documented [here](./gui.md).

The following code is taken from [the simpleplume example](../apps/simpleplume.py).

```python
class Simpleplume(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, 'Simpleplume', stride=5)
        smoke = world.Smoke(Domain([80, 64], SLIPPERY))
        world.Inflow(Sphere((10, 32), 5), rate=0.2)
        self.add_field('Density', lambda: smoke.density)
        self.add_field('Velocity', lambda: smoke.velocity)

Simpleplume().show()
```

By subclassing `FieldSequenceModel`, we inherit the functions

- `add_field` to make data visible in the GUI,
- `show` which launches the web service,
- `step` which calls `world.step()` by default but can be overriden to implement custom behaviour.

Slightly more complex examples can be found in 
[smokedemo.py](../apps/smokedemo.py) which adds obstacles to the scene and 
[movementdemo.py](../apps/movementdemo.py) which moves the inflow around.


### Running on the GPU

For GPU execution, TensorFlow needs to be installed (see the [installation instructions](./install.md)).
To run the simulation using TensorFlow, change the first line `from phi.flow import *` to `from phi.tf.flow import *`.
This replaces the imported `FieldSequenceModel` with a TensorFlow-enabled version.
This new `FieldSequenceModel` bakes the physics into a TensorFlow graph in `show()` before the GUI is launched.


