
# Fluid Simulation


## Design principles and Core Classes

The core classes are `FluidSimulation` and `StaggeredGrid` which are located in [phi.flow](../phi/flow/flow.py) and [phi.math.nd](../phi/math/nd.py), respectively.

```python
from phi.flow import FluidSimulation, StaggeredGrid
sim = FluidSimulation([64, 64])
```

Φ-*Flow* is dimensionality-agnostic, meaning it can run in any Euclidean space. The dimensions are passed to the `FluidSimulation` in the constructor.

Once a simulation object is created, it can be used to create the required fields.

```python
velocity = sim.placeholder("staggered", "velocity)
density = sim.placeholder(1, "density")
initial_velocity = sim.zeros("staggered")
initial_density = sim.zeros()
```

Notice that in the above example, velocity and initial_velocity are instances of `StaggeredGrid`.

The above code can also be rewritten using Python's `with` statement:

```python
with FluidSimulation([64, 64]):
    velocity = placeholder("staggered", "velocity)
    density = placeholder(1, "density")
    initial_velocity = zeros("staggered")
    initial_density = zeros()
```

A typical simulation step could comprise of the following steps. The following code takes the state of a state simulation, density and velocity, to the next time step.

```python
density = velocity.advect(density) + inflow_density
velocity = divergence_free(velocity.advect(velocity) + buoyancy(density))
```

In detail, this performs the following steps:

- Advect the density using the current velocity
- Add the inflow density to the advected density
- Advect the velocity
- Add buoyancy force to the velocity, depending on the density
- Make the velocity field divergence-free by solving for pressure and subtracting the pressure divergence


### Staggered grids

Staggered grids are a key component of the marker and cell (MAC) method. They sample the velocity components at the centers of the corresponding lower faces of grid cells. This makes them fundamentally different from regular arrays or tensors which are sampled at the cell centers. Staggered grids make it easy to compute the divergence of a velocity field.

![](./Staggered.png)

In Φ-*Flow*, staggered grids are represented as instances of [StaggeredGrid](../phi/math/nd.py). They have one more entry in every spatial dimension than corresponding centered fields since the upper face of the upper most cell needs to be included as well.

Staggered grids can be created from the [FluidSimulation](../phi/flow/flow.py) or [TFFluidSimulation](../phi/tf/flow.py) object.

```python
# Create a StaggeredGrid
velocity = sim.placeholder("staggered")
velocity = sim.zeros("staggered")
```

They can also be created from centered fields.

```python
staggered_grad = StaggeredGrid.gradient(centered_field)
staggered_field_x = StaggeredGrid.from_scalar(centered_field, [0, 0, 1])
```

`StaggeredGrid`s can hold both TensorFlow tensors and NumPy `ndarray`s, depending on the simulation configuration. They support basic backend operations and can be passed to `TFSimulation.run()` like TensorFlow tensors.

Some useful operations include:

```python
advected_field = velocity.advect(field)
curl = velocity.curl()
divergence = velocity.divergence()
```

To get a tensor or ndarray object from a staggered grid, one of the following sampling methods can be used.

```python
velocity.staggered  # array containing staggered components
velocity.at_centers()  # Interpolated to cell centers
velocity.at_faces(axis)  # Interpolated to face centers of given axis index
```



### Pressure Solvers

Given a simulation and velocity field or divergence thereof, the corresponding pressure field can be calculated using
```python
pressure = sim.solve_pressure(velocity)
```
which is implicitly called by `sim.divergence_free(velocity)`.

By default, the solver, registered with the `FluidSimulation` upon construction, is used.
A custom solver can be set using
```python
sim = FluidSimulation(..., solver=my_solver)
```

Here is an overview of the pressure solvers implemented in Φ-*Flow*:

| Solver   | Package             | Device       | Dependencies    | Status                                             |
| ---------|---------------------|--------------|-----------------|----------------------------------------------------|
| SparseCG   |[phi.solver.sparse](../phi/solver/sparse.py)    | CPU/GPU      | SciPy           | Stable                                             |
| SparseSciPy|[phi.solver.sparse](../phi/solver/sparse.py)  | CPU          | SciPy           | Stable, no control over accuracy, no loop counter  |
| CUDA     |[phi.solver.cuda](../phi/solver/cuda.py)      | GPU          | TensorFlow      | Stable, no support for initial guess               |
| GeometricCG |[phi.solver.geom](../phi/solver/geom.py)   | CPU/GPU/TPU  |                 | Stable, limited boundary condition support         |
| Multigrid |[phi.solver.multigrid](../phi/solver/multigrid.py)|              |                 | Stable, best performance in absence of boundaries  |


All solvers provide a gradient function for TensorFlow, needed to back-propagate weight updates through the pressure solve operation.

*Which solver should I use?*

Φ-*Flow* auto-selects an appropriate solver if you don't specify one manually.
If you have no special requirements, that selection should be fine.
Nevertheless, here are some recommendations:

- If you're working exclusively on the CPU, SparseSciPy is the fastest single-grid solver but offers the least amount of control.

- For the GPU, CUDA is the fastest single-grid solver.

- If your grid size is larger than 100 in any dimension, Multigrid can reduce the amount of iterations required.
It can currently use the following solvers per level: SparseCG, GeometricCG.
The multigrid solver is not yet optimized for obstacles.

- If you want to run a small number of iterations only and require backpropagation, use SparseCG, setting `max_iterations` and `autodiff=True`.

- GeometricCG is the slowest implementation.
However, it is also the simplest implementation and the easiest to understand.
It's also the only solver that is compatible with TensorFlow's TPU support.

You can also write your own solver.
Simply extend the class `phi.solver.base.PressureSolver` and implement the method `solve(...)`.


## Functional-style offline simulation

It is recommended to run simulations within the FieldSequenceModel framework which provides a browser-based GUI.
See [the documentation](gui.md) for how to implement an app with vizualization.
All simulation functionality can, however, be used without making use of the higher-level framework.

The following code sets up a NumPy simulation and runs it for 100 frames:

```python
from phi.flow import *

sim = FluidSimulation([64, 64], "closed")
inflow_density = sim.zeros()
inflow_density[0, 8, 22:64-22, 0] = 1
velocity = sim.zeros("velocity")
density = inflow_density
for i in range(100):
    print("Computing frame %d" % i)
    density = velocity.advect(density) + inflow_density
    velocity = sim.divergence_free(velocity.advect(velocity) + sim.buoyancy(density))
```

Here is the same simulation using TensorFlow:

```python
from phi.tf.flow import *

sim = TFFluidSimulation([64, 64], "closed")
inflow_density = sim.zeros()
inflow_density[0, 8, 22:64-22, 0] = 1
velocity_data = sim.zeros("staggered")
density_data = inflow_density
density = sim.placeholder(1, "density")
velocity = sim.placeholder("staggered", "velocity")
next_density = velocity.advect(density) + inflow_density
next_velocity = sim.divergence_free(velocity.advect(velocity) + sim.buoyancy(next_density))
for i in range(100):
    print("Computing frame %d" % i)
    density_data, velocity_data = sim.run([next_density, next_velocity], {density: density_data, velocity:velocity_data})
```

In this example, the TensorFlow graph only computes one frame of the simulation and is invoked multiple times. With this approach, the result of each frame computation is converted to NumPy arrays on the CPU.
It is also possible to build a graph with multiple frames but then the memory requirements and graph complexity scale with the number of time steps.

In any case, `sim.run(...)` needs to be called to execute the graph. This method extends TensorFlow's `session.run(...)` and can handle `StaggeredGrid`s.

## Boundary Conditions

Boundary conditions always exist at the borders of the domain and can also be enforced within the domain.

To specify what boundary conditions should be used at the borders, use the second argument of the FluidSimulation constructor. It can take the values `"open"`, `"closed"` or a `DomainBoundary` object specifying the boundary conditions for each face of the bounding box.

Boundary conditions within the domain usually correspond to obstacles that prevent fluid from flowing in or out of it. Obstacles can be added to the simulation using the `set_obstacle` method.
It can be used in two ways:

```python
sim.set_obstacle(box_size, origin)  # create a box obstacle from origin to origin+size
sim.set_obstacle(mask, origin)  # create an obstacle with a custom shape
```

To enable obstacles with a TensorFlow simulation, set `force_use_masks=True` in the constructor of `TFFluidSimulation`.

Internally, boundary conditions are handled using an `active_mask` and a `fluid_mask` tensor. For the purpose of displaying them, use `sim.extended_active_mask` or `sim.run(sim.extended_active_mask)`.