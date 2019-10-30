# Structs

In the most general sense, structs are containers for structured data.
The struct API enables easy iteration and manipulation of structured data.

Each struct consists of a number of named entries, similar to a dict. For instances of Struct, the names equal the attribute names of the instance. For dicts, the entry names equal the keys. Lists, tuples and arrays use the index of an entry as its name (type `int`).

The following objects are structs:

- Subclasses of [`phi.math.struct.Struct`](../phi/math/struct.py)
- Lists
- Tuples
- Dicts containing strings as keys
- NumPy arrays with `dtype=numpy.object`



Structs differentiate between two kinds of entries:

- *Attributes* hold the main data and can also hold sub-structs.
- *Properties* store additional information but not other structs.
- All *others* are not affected by any struct operations.

Lists, tuples and dicts can only hold *attributes*.
Attributes and properties can be "changed" using the `copy_with` function.
Thereby, the struct isn't altered but rather a duplicate with the new values is created.
While structs can be mutable in principle, the struct interface does not allow for changing a struct.

Structs can contain sub-structs, resulting in a cycle-free tree structure.

To use struct functions, the struct module needs to be imported:
```python
from phi import struct
```

## Iterating over structs

The struct interface provides the function `map` which iterates over all *attributes* of a struct and optionally all sub-structs.

Assume we have a struct `data` with sub-structs containing only tensors.
We could perform a mathematical operation on all tensors like so:

```python
struct.map(lambda x: x*2, data)
```

This iterates over all attributes of `data` and recursively over all of its sub-structs.
If we only wanted to affect the tensors directly held by `data`, we could call:

```python
struct.map(lambda x: x*2, data, recursive=False)
```

Assume we `data` struct held only the shapes of tensors as tuples or lists.
When iterating over the shapes, they will be assumed to be structs themselves and `map` will instead iterate over the individual numbers in the shape.
To prevent this, we need to declare the shapes as leaves of the struct tree. This can be achieved as follows:

```python
def is_shape(obj):
    return ...
struct.map(lambda shape: np.zeros(shape), data, leaf_condition=is_shape)
```

In some cases we require additional information when mapping a struct; not just the value but also where it is stored.
When calling `map(.., trace=True)`, a [`Trace`](../phi/math/struct.py) object is passed to the mapping function `f` instead of the value. In addition to retrieving the value via `trace.value`, it provides access to the attribute key as `trace.key` and the parent structs via `trace.parent`.


## Validitiy

As structs are supposed to hold data in a specific structure, there is a preferred data type for each entry.
For a CenteredGrid, the `data` attribute should be a tensor or array with a certain rank and the `velocity` of a `Smoke` object should be a `StaggeredGrid`.

An entry is _valid_ if its value if of the preferred data type.
Subclasses of `Struct` can implement validity checks and modify their entries to make them valid.

This hierarchy is not always needed, however. Many math functions return invalid structs such as `math.staticshape(obj)` which returns a struct containing shapes instead of data.
Code dealing with invalid structs should always be enclosed in a `with struct.anytype():` block.
This context skips all data validation steps.


## Implementing a custom struct

The following code snippet defines a custom `Struct`

```python
from phi import struct

class MyStruct(Struct):
    __struct__ = StructInfo(attributes=['_a'], properties=['_p'])

    def __init__(self, a, b):
        self._a = a
        self._p = p
        self.__validate__()

    @property
    def a():
        return self._a

    @property
    def p():
        return self._p

    def __validate_p__(self):
        self._p = str(self._p)
```

The line `__struct__ = StructInfo(attributes=['_a'], properties=['_p'])` must be present in every subclass of `phi.math.struct.Struct`.
It declares the public attributes and properties which can be accessed using `copied_with()`.

Structs are usually immutable which is why we declared all variables with a leading underscore and added read-only properties.
The method `def __validate_x__(self):` is called whenever the entry `x` changes.
In this case, the property `p` is supposed to hold a string so we convert it to a string whenever it is set. Calling `self.__validate__()` at the end of the constructor indirectly invokes all `__validate_x__` methods.
Validation methods are not called in the presence of a `with struct.anytype():` block.


## Usages in Φ<sub>*Flow*</sub>

In Φ<sub>*Flow*</sub>, structs are mostly used to store simulation states, i.e.
each attribute holds a tensor such as density or velocity of a smoke simulation.
In particular, the state base class [`phi.physics.physics.State`](../phi/physics/physics.py) extends `Struct`.

Properties are used to hold additional parameters for the simulation that should be included in the `description.json` file. Typical examples of these include viscosity or buoyancy_factor.

### Tensor initialization

Initializer functions such as `zeros` or `placeholder` internally call their counterparts in NumPy or TensorFlow.
They can take 1D-tensors describing the shape as input but also support structs holding shapes.
The call `zeros(StaggeredGrid([1,65,65,2]))` will return a `StaggeredGrid` holding a NumPy array.

Some states simplify this even further by allowing a syntax like `SmokeState(density=zeros)` or `SmokeState(velocity=placeholder)`.

The `placeholder` and `variable` initializers also infer the name of the resulting tensors from the attribute names.

### Data I/O

The data writing and reading system accepts structs and automatically infers their names from the attributes.
See the [data documentation](Reading_and_Writing_Data.md).

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
