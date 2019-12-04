# NumPy / TensorFlow Execution

Φ<sub>*Flow*</sub> supports two backends for executing the simulation code:

- NumPy executes operations when the corresponding method is called. The operations return nd-arrays which reference data in memory.
- TensorFlow has an equivalent set of operations. Unless eager execution is enabled, these do not compute the result right away. Instead, they create a node (TF `Tensor`) in the computational graph. The actual execution happens when `Session.run()` is invoked.

Φ<sub>*Flow*</sub> provides an abstraction layer in [its math package](../phi/math) which dynamically decides which backend to use depending on the inputs.
All available methods are listed [here](../phi/math/base.py) and support any [structs](Structs.ipynb) as input..

If any input to a math method is a `Tensor`, the TensorFlow version of the method will be called which returns a `Tensor`. Only if all inputs are arrays or numbers will the output be a NumPy array.

## Simulating with TensorFlow

By default, all simulations are initialized with NumPy arrays, e.g. calling `Fluid(Domain([64, 64])` creates a NumPy array for the fluid marker density and the velocity field.

Consequently, when `step()` is called to execute the simulation, it produces new states containing NumPy arrays.

There is an easy way to execute a simulation with TensorFlow, however.
By calling `tf_bake_graph(world, session)`, a TensorFlow graph is created which performs a single simulation step when executed. Additionally, all physics objects in the world are replaced by invocations to this graph.
When using a `App`, this is performed automatically on the default `world` if `App` is imported from `phi.tf.flow`.

TensorFlow can also be used manually by replacing NumPy arrays with `Tensor`s.

```python
from phi.tf.flow import *

fluid1 = Fluid(Domain([64, 64]), density=placeholder, velocity=variable)
fluid2 = INCOMPRESSIBLE_FLOW.step(fluid1)
```

In this example, the `Fluid` object is initialized with a TensorFlow placeholder and a TensorFlow variable instead of NumPy arrays.
Consequently, `fluid2` holds nodes from the computational graph and can be passed to `Session.run()` as a fetch argument.
