# Pressure Solvers

A vital step of Eulerian fluid simulations is the pressure solve.
Φ<sub>*Flow*</sub> provides two low-level functions to compute the pressure: `solve_pressure` and `divergence_free`.
`solve_pressure` computes and returns the pressure from a velocity or divergence field while
`divergence_free` also updates the velocity field and returns a divergence-free field.
Which algorithm is used to compute the pressure can be specified using the `pressure_solver` argument which
both of these methods provide.

In a high-level setting, the solver is specified by the `Physics` object that handles the simulation.
For fluid, this defaults to the global object `INCOMPRESSIBLE_FLOW`.
If you want to use a custom solver, simply replace the physics with a new IncompressibleFlow or directly set the pressure_solver property.

```python
from phi.flow import *

INCOMPRESSIBLE_FLOW.pressure_solver = SparseCG(accuracy=1e-4, max_iterations=200)
# or
world.add(Fluid([32, 32]), physics=IncompressibleFlow(pressure_solver=...))
```

Here is an overview of the pressure solvers implemented in Φ<sub>*Flow*</sub>:

| Solver        | Package                                             | Device       | Dependencies    | Status                                             |
| --------------|-----------------------------------------------------|--------------|-----------------|----------------------------------------------------|
| `SparseCG`    | [phi.physics.pressuresolver.sparse](../phi/physics/pressuresolver/sparse.py)        | CPU/GPU      | SciPy           | Stable                                             |
| `SparseSciPy` | [phi.physics.pressuresolver.sparse](../phi/physics/pressuresolver/sparse.py)        | CPU          | SciPy           | Stable, no control over accuracy, no loop counter  |
| `CUDA`        | [phi.physics.pressuresolver.cuda](../phi/physics/pressuresolver/cuda.py)            | GPU          | TensorFlow      | Stable, no support for initial guess               |
| `GeometricCG` | [phi.physics.pressuresolver.geom](../phi/physics/pressuresolver/geom.py)            | CPU/GPU/TPU  |                 | Stable, limited boundary condition support         |
| `MultiscaleSolver`  | [phi.physics.pressuresolver.multigrid](../phi/physics/pressuresolver/multiscale.py) |              |                 | Stable, best performance in absence of boundaries  |

All solvers provide a gradient function for TensorFlow, needed to back-propagate weight updates through the pressure solve operation.

*Which solver should I use?*

Φ<sub>*Flow*</sub> auto-selects an appropriate solver if you don't specify one manually.
If you have no special requirements, that selection should be fine.
Nevertheless, here are some recommendations:

- If you're working exclusively on the CPU, `SparseSciPy` is the fastest single-grid solver but offers the least amount of control.

- For the GPU, `CUDA` is the fastest single-grid solver.

- If your grid size is larger than 100 in any dimension, `MultiscaleSolver` can reduce the amount of iterations required.
It can currently use the following solvers per level: `SparseCG`, `GeometricCG`.
The multigrid solver is not yet optimized for obstacles.

- If you want to run a small number of iterations only and require backpropagation, use `SparseCG`, setting `max_iterations` and `autodiff=True`.

- `GeometricCG` is the slowest implementation.
However, it is also the simplest implementation and the easiest to understand.
It's also the only solver that is compatible with TensorFlow's TPU support.

You can also write your own solver.
Simply extend the class `phi.physics.pressuresolver.base.PressureSolver` and implement the method `solve(...)`.
