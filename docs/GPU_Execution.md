
# Optimizing Performance
Simulation code can be accelerated in multiple ways.
When optimizing your code, you first have to understand where the time is being spent.
Φ<sub>Flow</sub> includes an easy-to-use profiler that tells you which operations and functions eat up your computation time.

Enabling GPU execution may speed up your code when dealing with large tensors,
especially when using the custom CUDA kernels.
Switching to Graph mode may also reduce computational overheads.

## Profiler
The integrated profiler can be used to measure the time spent on each operation, independent of which backend is being used.
Additionally, it traces the function calls to let you see which high-level function calls the operations belong to.

To profile a code block, use `backend.profile()`.
```python
from phi.flow import *

with backend.profile() as prof:
    simulation_step()
prof.save_trace('Trace.json')
prof.print(min_duration=1e-2)
```
This above code stores the full profile in the [trace event format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU)
and prints all operations and functions that take more than 10 ms (`min_duration=1e-2`) to the console.

To view the profile, open Google Chrome and go to the address `chrome://tracing/`.
Then drag the file `Trace.json` into the browser window.
There you may zoom your view and click on any block to view additional information.

## Enabling GPU Execution
All simulations based on Φ<sub>Flow</sub> can be computed on the GPU without transferring data back to the CPU.
This requires a GPU-enabled TensorFlow or PyTorch installation, see the [installation instructions](Installation_Instructions.md).

Moving computations to the GPU can greatly increase the performance of simulation code but there are also drawbacks.
Since GPUs have access to many more processors than a CPU, GPU operations finish much faster than CPU operations.
However, for each GPU operation, one or multiple CUDA kernels have to be launched which adds a significant overhead.
This overhead is almost independent of the involved tensor sizes, so the speedup is greatest for large tensors.

Therefore, your code should be vectorized as much as possible.
Instead of performing an action multiple times, stack the data along a [batch dimension](https://tum-pbs.github.io/PhiFlow/Math.html#shapes).
Φ<sub>Flow</sub> tensors support arbitrary numbers of named batch dimensions with no reshaping required.

To run your code with either TensorFlow or PyTorch, [select the corresponding backend](https://tum-pbs.github.io/PhiFlow/Math.html#backend-selection) by choosing one of the following imports:

- TensorFlow: `from phi.tf.flow import *`
- PyTorch: `from phi.torch.flow import *`

**Native CUDA Kernels** (*Experimental*).
Φ<sub>Flow</sub> comes with a number of CUDA kernels for TensorFlow that accelerate specific operations such as grid interpolation or solving linear systems of equations.
These GPU operators yield the best overall performance, and are highly recommended for larger scale simulations or training runs in 3D.
To use them, download the Φ<sub>Flow</sub> sources and compile the kernels, following the [installations instructions](Installation_Instructions.md).
PyTorch already comes with a fast GPU implementation of grid interpolation.

## Graph Compilation (JIT)
Φ<sub>Flow</sub> supports both static and dynamic execution.
In graph mode, execution is usually faster, but an additional overhead is required for setting up the graph.
Also, certain checks and optimizations may be skipped in graph mode.

There are two ways of compiling a static graph

* `trace_function()` (recommended): The functions
  [`phi.math.trace_function()`](https://tum-pbs.github.io/PhiFlow/phi/math/#phi.math.trace_function) and 
  [`phi.field.trace_function()`](https://tum-pbs.github.io/PhiFlow/phi/field/#phi.field.trace_function)
  use the backend-specific compiler, if available, to compile a static graph for `Tensor` or `Field`-valued functions, respectively.
* Backend compiler: You may trace or compile functions manually using PyTorch, Jax or TensorFlow.
  This involves manually getting all native tensors since backend compilers do not support `phi.math.Tensor` or `Field` arguments.

**Gradients.**
Computing gradients may be easier in graph mode since no special actions are required for recording the operations.
In eager execution mode, gradient recording needs to be enabled using one of the following methods:

1. Code within a `with math.record_gradients():` block will enable gradient recording for both TensorFlow and PyTorch.
2. For TensorFlow, a `GradientTape` may be used directly. Retrieve TensorFlow tensors to watch using `Tensor.native()`.
3. For PyTorch, the `requires_grad` attribute may be set to `True` manually. Retrieve PyTorch tensors using `Tensor.native()`.

Methods 2 and 3 require special handling for non-uniform tensors. Manually iterate over the contained uniform tensors using `tensor.<dim>.unstack()`
and watch each element using `native()`.