
# Structs

In the most general sense, structs are containers for data.
They enable iteration over its contents.

While structs can be mutable, the struct interface does not allow for changing a struct.
However, the contents can be swapped for other contents and reassembled into a new struct of the same type.

Typically, struct objects contain multiple tensor objects, such as one tensor for density and one for velocity.

Structs can contain sub-structs and the resulting struct hierarchy should not be altered.
The data contained in struct, however, is not fixed to a specific data type.
Instead, structs must support data of any type.


## Definition and Functionality

The following objects are structs:

- Subclasses of `phi.math.struct.Struct`
- Lists
- Tuples
- Dicts containing strings as keys

Subclasses of `phi.math.struct.Struct` must override the `disassemble` method
which returns a list of all contained attributes, including attributes of sub-structs, and a reassembly function.
Sub-structs themselves are not contained in the attribute list.

The function `disassemble(object)` (not called on an object) implements this functionality also for other objects
and should always be preferred over `struct.disassemble()`.

For a struct containing TensorFlow tensors or NumPy arrays, the tensors could be read like this:
```python
tensors, reassemble = disassemble(struct)
new_struct = reassemble(tensors)
```
The second line creates a duplicate of the original `struct`.
Note, that the elements of `tensors` can be altered or replaced before reassembly.
The function `shape(struct)` makes use of this to return a `struct` holding the shapes of tensors from the original struct.


## Usages in Φ<sub>*Flow*</sub>

All state-related objects in Φ<sub>*Flow*</sub> are compatible with the Struct interface.

In particular, the `State` class is a subclass of `Struct` while TensorFlow tensors and NumPy arrays are supported as well.


### Struct initialization and shapes

Shapes are the same structs

`struct = zeros(struct_shape)`


### Data I/O

All DataIterators build a Struct for each batch.

Use Strings or DataChannels instead of tensors.


### Session

`placeholder(struct_shape)`
`variable(struct_shape)`

Session.run takes a Struct as fetch argument.

The `feed_dict` supports placeholder structs for keys and value-structs for values.