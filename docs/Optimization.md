# Optimization and Linear Systems of Equations
The backends PyTorch, TensorFlow and Jax have built-in automatic differentiation functionality.
However, the respective APIs vary widely in how the gradients are computed.
Φ<sub>Flow</sub> seeks to unify optimization and gradient computation so that code written against the Φ<sub>Flow</sub> API will work with all backends.
Nevertheless, we recommend using backend-specific optimization for certain tasks like neural network training.

The following overview table shows which Φ<sub>Flow</sub> optimization functions are supported by which backends.

| Backend    | solve_linear | minimize | functional_gradient | NN Training |
|------------|--------------|----------|---------------------|-------------|
| PyTorch    | ✓            | ✓        |   ✓                 |      ✓         |
| TensorFlow | ✓            | ✓        |   ✓                 |      ✓         |
| Jax        | ✓            | ✓        |   ✓                 |                |
| NumPy      | ✓            |          |                     |                |


## Physics Optimization
Φ<sub>Flow</sub> provides functions for solving linear equations as well as finding optimal parameters for non-linear problems.
These functions work with all backends but run much slower with NumPy due to the lack of analytic gradients.

### Nonlinear Optimization
The function [`math.minimize()`](phi/math/#phi.math.minimize) and [`solve_nonlinear()`](phi/math/#phi.math.solve_nonlinear) solve unconstrained nonlinear optimization problems.
The following example uses L-BFGS-B to find a solution to a nonlinear optimization problem using the `phi.field` API:
```python
def loss(x1: Grid, x2: Grid) -> math.Tensor:
  return field.l2_loss(physics(x1, x2) - target)

x0 = x0_1, x0_2
solution = math.minimize(math.jit_compile(loss), math.Solve('L-BFGS-B', 0, 1e-3, x0=x0))
```


### Linear Equations
For solving linear systems of equations, Φ<sub>Flow</sub> provides the function [`math.solve_linear()`](phi/math/#phi.math.solve_linear).
The following example uses the conjugate gradient algorithm to solve_linear `f(x) = y`:
```python
@math.jit_compile_linear
def f(x: Grid) -> Grid:
    return field.where(mask, 2 * field.laplace(x), x)

x = math.solve_linear(f, y, math.Solve('CG', 1e-3, 0, x0=x0))
```

Which solver implementation is used, depends on the backend.
However, all backends support conjugate gradient (`'CG'`) and conjugate gradient with adaptive step size (`'CG-adaptive'`).
Specify `'auto'` to let Φ<sub>Flow</sub> chose an appropriate solver.

Overview: Implementations

| Method        | PyTorch | TensorFlow |                               Jax                              |                           NumPy                          |
|---------------|:-------:|:----------:|:--------------------------------------------------------------:|:--------------------------------------------------------:|
| 'CG'          |  Φ-Flow |   Φ-Flow   | `jax.scipy.sparse.linalg.cg` (only in jit mode, no trajectory) |         `scipy.sparse.linalg.cg` (no trajectory)         |
| 'CG-adaptive' |  Φ-Flow |   Φ-Flow   |                  Φ-Flow (only function-based)                  |                          Φ-Flow                          |
| 'auto'        |  Φ-Flow |   Φ-Flow   | `jax.scipy.sparse.linalg.cg` (only in jit mode, no trajectory) | `scipy.sparse.linalg.spsolve` (only for sparse matrices) |


### Handling Failed Optimizations
Both `solve_linear` and `minimize` return only the solution.
To access further information about the optimization, run the optimization within a [`SolveTape`](phi/math/#phi.math.SolveTape) context.

When a solve does not find a solution, a subclass of [`ConvergenceException`](phi/math/#phi.math.ConvergenceException) is thrown.
```python
solve = math.Solve('CG', 1e-3, 0, max_iterations=300)
try:
    solution = field.solve_linear(..., solve=solve)
    solution = field.minimize(..., solve=solve)
    converged = True
except NotConverged as not_converged:
    print(not_converged)
    last_estimate = not_converged.x
except Diverged as diverged:
    print(diverged)
iterations_performed = solve.result.iterations  # available in any case
```

### Backpropagation
Currently, backpropagation is supported for `solve_linear()`.
The following code shows how to specify different settings for the backpropagation solve:
```python
from phi.flow import *
gradient_solve = math.Solve('auto', 1e-5, 0, max_iterations=100, x0=x0)
solve = math.Solve('CG', 1e-5, 1e-5, gradient_solve=gradient_solve, x0=None)
```
Solves during backpropagation raise the same exceptions as in the forward pass.
The only exception is TensorFlow where a warning message is printed instead since exceptions are not supported during backpropagation.
Information about backpropagation solves can be obtained the same ways with the forward solves,
using a [`SolveTape`](phi/math/#phi.math.SolveTape) context around the gradient computation.


## Computing Gradients
There are two ways to evaluate gradients using Φ<sub>Flow</sub>:
computing gradients of functions and computing gradients of tensors.
The former is the preferred way and works will all backends except for NumPy.

### Gradients of functions
The functions
[`math.functional_gradient()`](phi/math/#phi.math.functional_gradient) and
[`field.functional_gradient()`](phi/field/#phi.field.functional_gradient)
compute the gradient of a Python function with respect to one or multiple arguments.
The following example evaluates the gradient of `physics` w.r.t. two of its inputs.
```python
def physics(x: Grid, y: Grid, target: Grid) -> math.Tensor:
    x, y = step(x, y)
    return field.l2_loss(y - target), x

gradient = field.functional_gradient(physics, wrt='x,y', get_output=True)

loss, x, (dx, dy) = gradient(...)
```
In the above example, `wrt=[0, 1]` specifies that we are interested in the gradient with respect to the first and second argument of `physics`.
With `get_output=True`, evaluating the gradient also returns all regular outputs of `physics`.
Otherwise, only `dx, dy` would be returned.

Note that the first output of `physics` must be a scalar `Tensor` object.
All other outputs are not part of the gradient computation.

Note that higher-order derivatives might not work as expected with some backends.


## Backend-specific Optimization
*Recommended for neural network training.*

Unfortunately all supported backends have a different approach to computing gradients:

* PyTorch: gradients are recorded per flagged tensor
* TensorFlow: gradients are recorded per flagged operation
* Jax: gradients are computed for functions

TensorFlow and PyTorch include various optimizers for neural network (NN) training.
Additionally, NN variables created through the respective layer functions are typically marked as variables by default,
meaning the computational graph for derived tensors is created automatically.
