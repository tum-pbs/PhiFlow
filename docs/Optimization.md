# Optimization and Training
The backends PyTorch, TensorFlow and Jax have built-in automatic differentiation functionality.
Unfortunately, the respective APIs vary widely in how the gradients are computes

* PyTorch: gradients are recorded per flagged tensor
* TensorFlow: gradients are recorded per flagged operation
* Jax: gradients are computed for functions

Φ<sub>Flow</sub> provides unified methods for computing gradients.
However, depending on the application, it may be easier to use the backend functionality directly.


## Backend-specific Optimizers
*Recommended for neural network training.*

TensorFlow and PyTorch include various optimizers for neural network (NN) training.
Additionally, NN variables created through the respective layer functions are typically marked as variables by default,
meaning the computational graph for derived tensors is created automatically.

The following steps are required to use these optimizers with Φ<sub>Flow</sub>:

* Load data
* Get NN input tensors with correct dimension order using `Tensor.native('batch,vector,x,y')`
* Run NN
* Convert output to Φ `Tensor` or `Field` using `math.wrap()` or `Domain.xx_grid()`
* Run physics simulation
* Define loss and get backend value using `loss.native()`
* Compute gradients w.r.t. NN and update weights with optimizer

A PyTorch example of this can be seen in the demo
[network_training_pytorch.py](https://github.com/tum-pbs/PhiFlow/blob/master/demos/network_training_pytorch.py).

For TensorFlow, a `GradientTape` context is required around network evaluation, physics and loss definition.


## Unified Automatic Differentiation
Φ<sub>Flow</sub> provides two paradigms for computing gradients.

* **Gradients of functions** (*PyTorch, TensorFlow, Jax*): The functions
  [`math.functional_gradient()`](phi/math/#phi.math.functional_gradient) and
  [`field.functional_gradient()`](phi/field/#phi.field.functional_gradient)
  compute the gradient of a Python function with respect to one or multiple arguments.
  *Warning:* Higher-order derivatives might not work as expected with some backends.
* **Gradients of tensors** (*PyTorch, TensorFlow*): Operations within a
  [`math.record_gradients`](phi/math/#phi.math.record_gradients)
  context record gradient information.
  Calling [`math.gradients`](phi/math/#phi.math.gradients)
  then computes the gradients with respect to a previously marked tensor.
  *Warning*: You have to manually detach PyTorch tensors that will be used outside the context.
  Higher-order derivatives are not supported for PyTorch.


## SciPy Optimizers
See
[`math.minimize()`](phi/math/#phi.math.minimize) and
[`math.solve()`](phi/math/#phi.math.solve).
