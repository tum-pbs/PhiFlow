# What to avoid in Φ<sub>Flow</sub>

The feature sets of PyTorch, Jax and TensorFlow vary, especially when it comes to function operations.
This document outlines what should be avoided in order to keep your code compatible with all backends.


## Limitations of `jit_compile`

Do not do any of the following inside functions that may be jit-compiled.

### Avoid side effects (Jax)
Jit-compiled functions should be [pure](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions).
Do not let any values created inside a jit-compiled function escape to the outside.


### Avoid nested functions and lambdas (Jax)
Do not pass temporary functions to any custom-gradient function.
Temporary functions are those whose `id` changes each time the function to be jit-compiled is called.
In particular, do not `solve_linear` temporary function or pass temporary functions as `preprocess_y`.


### Avoid nested `custom_gradient` (PyTorch)
Functions that define a custom gradient via `math.custom_gradient` should not call other custom-gradient functions.
This may result in errors when jit-compiling the function.


### Do not jit-compile neural networks (Jax)
Do not run neural networks within jit-compiled functions.
The only exception is the `loss_function` passed to `update_weights()`.
This is because Jax requires all parameters including network weights to be declared as parameters but Φ<sub>Flow</sub> does not.


### Do no compute gradients (PyTorch)
Do not call `math.functional_gradient` within a jit-compiled function.
PyTorch cannot trace backward passes.


### Do not use `SolveTape` (PyTorch)
`SolveTape` does not work while tracing with PyTorch.
This is because PyTorch does not correctly trace `torch.autograd.Function` instances which are required for the implicit backward solve.


## Memory Leaks

Memory leaks can occur when transformed function are repeatedly called with non-compatible arguments.
This can happen with `custom_gradient` but also `jit_compile`, `functional_gradient` or `hessian`.
Each time such a function is called with new keyword arguments or tensors of new shapes, a record is stored with that function.
For top-level functions, such as `solve_linear`, that record will be held indefinitely.
In cases, where this becomes an issue, you can manually clear the these records or `jit_compile` the function producing the repeated calls.

To clear the cached mappings of a transformed function `f`, use
```python
f.traces.clear()
f.recorded_mappings.clear()
```
