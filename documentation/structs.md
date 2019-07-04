
# Structs

In the most general sense, structs are containers for data.
They enable iteration over arbitrary structures of data.

The following objects are structs:

- Subclasses of [`phi.math.struct.Struct`](../phi/math/struct.py)
- Lists
- Tuples
- Dicts containing strings as keys

Structs differentiate between two kinds of entries:
- *Attributes* hold the main data and can also hold sub-structs.
- *Properties* store additional information but not other structs.
- All *others* are not affected by any struct operations.

Lists, tuples and dicts can only hold *attributes*.
Attributes and properties can be "changed" using the `copy_with` function.
Thereby, the struct isn't altered but rather a duplicate with the new values is created.
While structs can be mutable in principle, the struct interface does not allow for changing a struct.

Structs can contain sub-structs, resulting in a cycle-free tree structure.

The data contained in struct is not fixed to a specific data type.
Structs must support data of any type at creation time.


## Iterating over structs

The struct interface provides the function `map` which iterates over all *attributes* of a struct and optionally all sub-structs.

Assume we have a struct `data` with sub-structs containing only tensors.
We could perform a mathematical operation on all tensors like so:
```python
struct.map(lambda x: x*2, data)
```
This iterates over all attributes of `data` and recursively over all of its sub-structs.
If we only wanted to affect the tensors directly held by `data`, we could call
```python
struct.map(lambda x: x*2, data, recursive=False)
```

Assume we `data` struct held only the shapes of tensors as tuples or lists.
When iterating over the shapes, they will be assumed to be structs themselves and `map` will instead iterate over the individual numbers in the shape.
To prevent this, we need to declare the shapes as leaves of the struct tree. This can be achieved as follows:
```python
def if_shape(obj):
    return ...
struct.map(lambda shape: np.zeros(shape), data, leaf_condition=is_shape)
```

In some cases we require additional information when mapping a struct; not just the value but also where it is stored.
When calling `map(.., trace=True)`, a [`Trace`](../phi/math/struct.py) object is passed to the mapping function `f` instead of the value. In addition to retrieving the value via `trace.value`, it provides access to the attribute key as `trace.key` and the parent structs via `trace.parent`.

## Implementing a custom struct


Subclasses of `phi.math.struct.Struct` must declare their attributes and properties using the class variable `__struct__`, e.g.

```python
class MyStruct(Struct):
    __struct__ = StructInfo(attributes=('_a',), properties=('_p'))

    @property
    def a():
        return self._a

    @property
    def p():
        return self._p
```




## Usages in Φ<sub>*Flow*</sub>

In Φ<sub>*Flow*</sub>, structs are mostly used to store simulation states, i.e.
each attribute holds a tensor such as density or velocity of a smoke simulation.
In particular, the state base class [`phi.physics.physics.State`](../phi/physics/physics.py) extends `Struct`.

Properties are used to hold additional parameters for the simulation that should be included in the `description.json` file. Typical examples of these include viscosity or buoyancy_factor.


### Tensor initialization

Initializer functions such as `zeros` or `placeholder` internally call their counterparts in NumPy or TensorFlow.
They can take 1D-tensors describing the shape as input but also support structs holding shapes.
The call `zeros(StaggeredGrid([1,65,65,2]))` will return a `StaggeredGrid` holding a NumPy array.

Some states simplify this even further by allowing a syntax like `SomkeState(density=zeros)` or `SmokeState(velocity=placeholder)`.

The `placeholder` and `variable` initializers also infer the name of the resulting tensors from the attribute names.


### Data I/O

The data writing and reading system accepts structs and automatically infers their names from the attributes.
See the [data documentation](data.md).


### Session

The [`Session`](../phi/tf/session.py) class is a customized version of `tf.Session` which accepts structs for the `fetches` argument as well as inside the `feed_dict`.

This can be used to quickly run states through a graph like so:
```python
numpy_state = Smoke(density=zeros, velocity=zeros)
placeholder_state = Smoke(density=placeholder, velocity=placeholder)
output_state = SMOKE.step(placeholder_state)
session = Session(scene)
numpy_state = session.run(output_state, {placeholder_state: numpy_state})
```