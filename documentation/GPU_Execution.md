
# Accelerated Simulations on the GPU

Simulations run in Φ<sub>Flow</sub> can be computed fully on the GPU.

Requirements
- [TensorFlow](https://www.tensorflow.org/) with GPU support or [PyTorch](https://pytorch.org/) with GPU support
- CUDA, cuDNN (matching TensorFlow / PyTorch distribution)

The preferred way to run simulations on the GPU is using TensorFlow 1.14 or 1.15 with Python 3.6.


## Native CUDA Kernels

Φ<sub>Flow</sub> comes with a number of TensorFlow CUDA kernels that accelerate specific operations such as the pressure / Poisson solve and the advection step.
These GPU operators yield the best overall performance, and are highly recommended for larger scale simulations or training runs in 3D.
To use them, download the Φ<sub>Flow</sub> sources and compile the kernels, following the [installations instructions](Installation_Instructions.md).

To use the CUDA pressure solver, pass `pressure_solver=CUDASolver()` when creating `IncompressibleFlow` or pass the solver directly to a function that requires a solve, such as `poisson_solve()`, `solve_pressure()` or `divergence_free()`.


## Running the demos on the GPU

Most demos, such as [simpleplume.py](../demos/simpleplume.py) use the `App` class to progress the simulation.
Running these demos on the GPU is as simple as exchanging the import:

- Standard NumPy (CPU) import: `from phi.flow import *`
- TensorFlow (CPU/GPU) import: `from phi.tf.flow import *`
- PyTorch (CPU/GPU) import : `from phi.torch.flow import *`

This works because each package defines its own `App` class.

The TensorFlow version of `App` creates a static TensorFlow graph representing one time step of the `world` physics (can be disabled using `app.auto_bake = False`).
The data stored in `world.state` are still NumPy arrays.

The PyTorch version of `App` converts all data to PyTorch tensors (can be disabled using `app.auto_convert = False`).
The data stored in `world.state` are PyTorch tensors.


## Setting up a TensorFlow / PyTorch Simulation

Φ<sub>Flow</sub> gives you full control over which operations should be performed by TensorFlow or PyTorch.
You can even have part of the simulation be computed with NumPy and another run on the GPU.

This is because Φ<sub>Flow</sub> determines the appropriate computing library (NumPy, TensorFlow or PyTorch) for each atomic operation (`phi.math` function) separately.
If all arguments are NumPy compatible, the corresponding NumPy call is made.
However, if there is at least one TensorFlow or PyTorch tensor involved, these libraries will be used instead of NumPy, converting NumPy arrays to tensors in the process.

*Example:* We create a fluid simulation using a TensorFlow tensor for the density
```python
from phi.tf.flow import *

fluid = Fluid([64, 40], density=placeholder)
```
Since we have not specified the velocity, it defaults to zero and will be initialized with a NumPy array.
When we do computations involving the density, TensorFlow functions will be called and the outputs will be TensorFlow tensors.
But if we only require the velocity for certain computations, functions from NumPy will be invoked.
The result of `IncompressibleFlow().step(fluid)` will therefore have TensorFlow tensors for both `density` and `velocity` since they mix in the simulation.


## Static and Dynamic Graphs

Φ<sub>Flow</sub> supports both static and dynamic graphs.
More precisely, each atomic operation (`phi.math` function) can be run in eager or non-eager mode and will behave exactly like you would expect from TensorFlow or PyTorch functions.

When using an `App`, the preferred way of TensorFlow execution are static graphs. Φ<sub>Flow</sub> will disable TensorFlow's eager execution by default.
