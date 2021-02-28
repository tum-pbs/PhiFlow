# Math Dictionary
This cheat sheet lists the Φ<sub>Flow</sub> analogues to specific functions from NumPy, TensorFlow, PyTorch and Jax.
Most [Φ<sub>Flow</sub> math functions](phi/math/index.html) are identical in name to their backend counterparts.
Only differing names are listed here.


## NumPy

| NumPy                       | Φ<sub>Flow</sub>              |
|-----------------------------|----------------------------------------|
| `class ndarray`             | `class Tensor`              |
| `class dtype`               | `class DType`              |
| `array(t)`                  | `tensor(t)`              |
| `t.ndim`                    | `t.rank`              |
| `t.astype(dtype)`           | `cast(t, dtype)`, `to_float(t)`, `to_int(t)`, `to_complex(t)` |
| `fft2()`, `rfft()`          | `fft()` |
| `tile(t, reps)`             | `expand(t, dim, size)`     |
| `reshape(t, s)`             | `join_dimensions(t, dims, dim)`, `split_dimension(t.dim, dims)`     |
| `concatenate()`             | `concat()`     |
| `t[...,::-1,...]`           | `t.dim.flip()`     |
| `random.random()`           | `random_uniform()`     |
| `random.standard_normal()`  | `random_normal()`     |
| `argwhere()`                | `nonzero()`     |
| `tensordot()`               | `dot()`     |
| `values[indices]`           | `gather(values, indices)`     |
| `t[indices] = values`           | `t + scatter(indices, values)`     |
| `t[bool_mask]`              | `boolean_mask(x, bool_mask)`     |


## TensorFlow

| TensorFlow                  | Φ<sub>Flow</sub>              |
|-----------------------------|----------------------------------------|
| `GradientTape()`            | `record_gradients()`                   |
| `class dtype`               | `class DType`              |
| `@function`                 | `trace_function`                   |
| `tile(t, reps)`             | `expand(t, dim, size)`     |
| `reshape(t, s)`             | `join_dimensions(t, dims, dim)`, `split_dimension(t.dim, dims)`     |
| `device_lib.list_local_devices()` | `TF_BACKEND.list_devices()`     |
| `executing_eagerly()`       | `all_available()`     |
| `equal(t1, t2)`             | `t1 == t2`     |
| `t[...,::-1,...]`               | `t.dim.flip()`     |
| `tensordot()`               | `dot()`     |
| `scatter_nd()`               | `scatter()`     |


## PyTorch

| PyTorch                     | Φ<sub>Flow</sub>               |
|-----------------------------|----------------------------------------|
| `class dtype`               | `class DType`              |
| `from_numpy(t)`             | `tensor(t, convert=True)`              |
| `t.detach()`                | `stop_gradient(t)`                     |
| `unsqueeze(t)`              | `expand_batch(t)`, `expand_spatial(t)`, `expand_spatial(t)` |
| `t.backward()`              | `gradients(t)` |
| `jit.trace(f)`              | `trace_function(f)` |
| `t.repeat(sizes)`           | `expand(t, dim, size)`     |
| `reshape(t, s)`             | `join_dimensions(t, dims, dim)`, `split_dimension(t.dim, dims)`     |
| `cuda.device_count()` <br /> `cuda.get_device_properties()`  | `TORCH_BACKEND.list_devices()`     |
| `t.permute()`               | `transpose(t)`     |
| `cat()`                     | `concat()`     |
| `t.flip(dim)`               | `t.dim.flip()`     |
| `masked_select()`           | `boolean_mask()`     |
| `sparse.FloatTensor`        | `TORCH_BACKEND.sparse_tensor()`     |
| `autograd.grad()`           | `gradients()`     |
| `t.requires_grad = True`    | `with record_gradients(t)`     |


## Jax
Most Jax functions are named after their NumPy counterparts. Please refer to the NumPy section for these.

| Jax                         | Φ<sub>Flow</sub>               |
|-----------------------------|----------------------------------------|
| `jit(f)`                    | `trace_function(f)`              |
| `grad(f)`                   | `gradient_function(f)`     |
| `devices()`                 | `JAX_BACKEND.list_devices()`     |

