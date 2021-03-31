# Optimization and Training
The backends PyTorch, TensorFlow and Jax have built-in automatic differentiation functionality.
However, the respective APIs vary widely in how the gradients are computed.
Φ<sub>Flow</sub> seeks to unify optimization and gradient computation so that code written against the Φ<sub>Flow</sub> API will work with all backends.
Nevertheless, we recommend using backend-specific optimization for certain tasks like neural network training.

The following overview table shows which Φ<sub>Flow</sub> optimization functions are supported by which backends.

| Backend    | solve | minimize | functional_gradient | record_gradients | NN Training |
|------------|-------|----------|---------------------|------------------|-------------|
| PyTorch    | ✓     | ✓        |   ✓                 |    ✓             |   ✓         |
| TensorFlow | ✓     | ✓        |   ✓                 |    ✓             |   ✓         |
| Jax        | ✓     | ✓        |   ✓                 |                  |             |
| NumPy      | ✓     | ✓        |                     |                  |             |


## Physics Optimization
Φ<sub>Flow</sub> provides functions for solving linear equations as well as finding optimal parameters for non-linear problems.
These functions work with all backends but run much slower with NumPy due to the lack of analytic gradients.

### Nonlinear Optimization
The functions [`math.minimize()`](phi/math/#phi.math.minimize) and [`field.minimize()`](phi/field/#phi.field.minimize)
solve unconstrained nonlinear optimization problems.
The following example uses L-BFGS-B to find a solution to a nonlinear optimization problem using the `phi.field` API:
```python
def loss(x: Grid) -> math.Tensor:
  return field.l2_loss(physics(x) - target)

solution = field.minimize(loss, x0, math.Solve('L-BFGS-B', 0, 1e-3))
```

### Linear Equations
For solving linear systems of equations, Φ<sub>Flow</sub> provides the functions
[`math.solve()`](phi/math/#phi.math.solve) and [`field.solve()`](phi/field/#phi.field.solve).
The following example uses the conjugate gradient algorithm to solve `A(x) = y`:
```python
@field.linear_function
def A(x: Grid) -> Grid:
  return field.where(mask, 2 * field.laplace(x), x)

x = field.solve(A, y, x0, math.Solve('CG', 1e-3, 0))
```
Solve can also be used to find solutions to nonlinear equations.
This is equivalent to minimizing the squared error with `minimize()`.

### Handling Failed Optimizations
Both `solve` and `minimize` return only the solution.
Further information about the optimization is stored in `solve.result` of the passed [`Solve`](phi/math/#phi.math.Solve) object.
When a solve does not find a solution, a subclass of
[`ConvergenceException`](phi/math/#phi.math.ConvergenceException) is thrown.
```python
solve = math.Solve('CG', 1e-3, 0, max_iterations=300)
try:
  solution = field.solve(..., solve_params=solve)
  solution = field.minimize(..., solve_params=solve)
  converged = True
except NotConverged as not_converged:
  print(not_converged)
  last_estimate = not_converged.x
except Diverged as diverged:
  print(diverged)
iterations_performed = solve.result.iterations  # available in any case
```

### Backpropagation
Currently, backprop is only supported by linear solves with the `'CG'` optimizer.
The following code shows how to specify different settings for the backprop solve:
```python
from phi.flow import *
gradient_solve = math.Solve('CG', 1e-4, 0, max_iterations=100)
solve = math.Solve('CG', 1e-5, 0, gradient_solve=gradient_solve)
```
Solves during backprop raise the same exceptions as in the forward pass.
Information about backprop solves can be obtained through `solve.gradient_solve.result`.


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

gradient = field.functional_gradient(physics, wrt=[0, 1], get_output=True)

loss, x, dx, dy = gradient(...)
```
In the above example, `wrt=[0, 1]` specifies that we are interested in the gradient with respect to the first and second argument of `physics`.
With `get_output=True`, evaluating the gradient also returns all regular outputs of `physics`.
Otherwise, only `dx, dy` would be returned.

Note that the first output of `physics` must be a scalar `Tensor` object.
All other outputs are not part of the gradient computation.

Note that higher-order derivatives might not work as expected with some backends.


### Gradients of Tensors
This method is only supported by PyTorch and TensorFlow.
Operations within a [`math.record_gradients`](phi/math/#phi.math.record_gradients)
context record gradient information.
Calling [`math.gradients`](phi/math/#phi.math.gradients)
then computes the gradients with respect to a previously marked tensor.

*Warning*: You have to manually detach PyTorch tensors that will be used outside the context.
Higher-order derivatives are not supported for PyTorch.


## Backend-specific Optimization
*Recommended for neural network training.*

Unfortunately all supported backends have a different approach to computing gradients:

* PyTorch: gradients are recorded per flagged tensor
* TensorFlow: gradients are recorded per flagged operation
* Jax: gradients are computed for functions

TensorFlow and PyTorch include various optimizers for neural network (NN) training.
Additionally, NN variables created through the respective layer functions are typically marked as variables by default,
meaning the computational graph for derived tensors is created automatically.

### PyTorch Neural Network
The following script shows how a PyTorch neural network can be trained.
See the demo [network_training_pytorch.py](https://github.com/tum-pbs/PhiFlow/blob/master/demos/network_training_pytorch.py)
for a full example.
```python
net = u_net(2, 2)
optimizer = optim.Adam(net.parameters(), lr=1e-3)

for training_step in range(100):
    data: Grid = load_training_data(training_step)
    optimizer.zero_grad()
    prediction: Grid = field.native_call(net, data)
    simulation_output: Grid = simulate(prediction)
    loss = field.l2_loss(simulation_output)
    loss.native().backward()
    optimizer.step()
```
In the above example, [`field.native_call()`](phi/field/#phi.field.native_call)
extracts the field values as PyTorch tensors with shape `(batch_size, channels, spatial...)`,
then calls the network and returs the result again as a `Field`.

Since `loss` is a `phi.math.Tensor`, we need to invoke `native()` to call PyTorch's `backward()` function.

### TensorFlow Neural Network
For TensorFlow, a `GradientTape` context is required around network evaluation, physics and loss definition.

