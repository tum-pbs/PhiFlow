# Math

The `phi.math` module \[[source](../phi/math)\] provides abstract access to tensor operations.
It internally uses NumPy, TensorFlow or PyTorch to execute the actual operations, depending on which backend is selected.
This ensures that code written against `phi.math` functions produces equal results on all backends.

To that end, `phi.math` provides a new `Tensor` class which should be used instead of directly accessing native tensors from NumPy, TensorFlow or PyTorch. 
While similar to the native tensor classes, `phi.math.Tensor`s have named and typed dimensions.

When performing operations such as `+, -, *, /, %, **` or calling `math` functions on `Tensor`s, dimensions are matched by name and type.
This eliminates the need for manual reshaping or the use of singleton dimensions.

Example:
```
from phi import math

>>> math.ones(x=5) + math.ones(x=5)
(x=5) float32  2.0 < ... < 2.0

>>> math.ones(x=5) + math.ones(batch=10)
(batch=10, x=5) float32  2.0 < ... < 2.0
```


## Shapes

The shape of `Tensor`s is represented by a `Shape` object which can be accessed as `tensor.shape`.
In addition to the integer sizes of the dimensions, the shape also stores the names of the dimensions as strings as well as their types.

There are three types of dimensions

* **Batch** dimensions are ignored by most operations. They are automatically added as needed.
* **Spatial** dimensions are associated with physical space. If two `Tensors`s live in different physical spaces, operations may raise `IncompatibleShapes` errors.
* **Channel** dimensions typically list vector elements or feature maps. They are automatically added as needed.

The preferred way to define a `Shape` is via the `shape()` function.
It takes the dimension sizes as keyword arguments.

```
>>> math.shape(batch=10, y=2, x=4, vector=2)
(batch=10, y=2, x=4, vector=2)
```

The dimension types are inferred from the names according to the following rules:

* Single letter -> Spatial dimension
* Starts with 'vector' -> Channel dimension
* Else -> Batch dimension

```
>>> math.shape(batch=10, y=2, x=4, vector=2).types
('batch', 'spatial', 'spatial', 'channel')
```

`Shape` objects should be considered immutable.
They provide many methods to effectively work with these dimensions.

Important `Shape` properties:

* `.sizes: tuple` enumerates the sizes as ints or None, similar to NumPy's shapes
* `.names: tuple` enumerates the dimension names
* `.rank: int = len(shape)` number of dimensions
* `.batch: Shape` / `.spatial: Shape` / `.channel: Shape` contains only batch / spatial / channel dimensions
* `.non_batch: Shape` / `.non_spatial: Shape` / `.non_channel: Shape` contains only the other two types of dimensions
* `.batch_rank: Shape` / `.spatial_rank: Shape` / `.channel_rank: Shape` alias for `.batch.rank`
* `.volume` number of elements a tensor of this shape contains

Important `Shape` methods:

* `get_size(name)` returns the size of a dimension
* `index(name)` returns the position of a dimension
* `combined(Shape)` returns a shape containing the dimensions from both shapes, equal to `&` operator
* `expand_<type>(size, name, pos)` adds a dimension to the shape
* `extend(Shape)` adds all dimensions from the other shape to this one
* `without(dims)` drops the specified dimensions
* `only(dims)`, `select(*names)` drops all other dimensions

Additional tips and tricks

* `tuple(shape)` is equal to `shape.sizes`.
* `'x' in shape` tests whether a dimension by the name of 'x' is present.
* `shape1 == shape2` tests equality including names, types and order of dimensions.
* `shape1 + shape2` adds the sizes of the shapes.
* `shape 1 & shape2` combines the shapes.
* `shape.x` returns the size of the dimension 'x'.


## Tensor creation

The `tensor()` function converts list, NumPy array or TensorFlow/PyTorch tensor to a `Tensor`.
The dimension names can be specified using the `names` keyword and dimension types are inferred from the names.
Otherwise, they are determined automatically.
Singleton batch dimensions are discarded.

```
>>> math.tensor([1, 2, 3])
(vector=3) int32  1, 2, 3

>>> math.tensor(numpy.zeros([1, 5, 4, 2]))
(x=5, y=4, vector=2) float64  0.0 < ... < 0.0

>>> math.tensor(numpy.zeros([1, 5, 1, 2]))
(x=5, y=1, vector=2) float64  0.0 < ... < 0.0

>>> math.tensor(numpy.zeros([3, 3, 1]), names=['y', 'x', 'time'])
(y=3, x=3) float64  0.0 < ... < 0.0
```

There are a couple of functions in the `phi.math` module for creating basic `Tensor`s.

* `zeros()`
* `ones()`
* `random_normal()`
* `random_uniform()`
* `meshgrid()`

Most functions allow the shape of the tensor to be specified via a `Shape` object or alternatively through the keyword arguments.

In the latter case, the dimension types are inferred from the names.



Examples

```
>>> math.zeros(x=5, y=4)  # Tensor with two spatial dimensions
(x=5, y=4) float32  0.0 < ... < 0.0

>>> math.zeros(math.shape(y=4, x=5))
(y=4, x=5) float32  0.0 < ... < 0.0

>>> math.meshgrid(x=5, y=(0, 1, 2))
(x=5, y=3, vector=2) int32  0 < ... < 4
```


## Indexing, slicing, unstacking

The recommended way of indexing or slicing tensors is using the syntax `tensor.<dim name>[start:end:step]`.
This can be chained to index multiple dimensions, e.g. `tensor.x[:2].y[1:-1]`.

Alternatively tensors can be indexed using a dictionary of the form `tensor[{dim: slice or int}]`.

Tensors can be unstacked along any dimension using `t.unstack(dim)` or `t.dim.unstack()`.
When passing the dimension size to the latter, tensors can even be unstacked along dimensions they do not posess.

```
>>> math.zeros(x=4).x.unstack()
(0.0, 0.0, 0.0, 0.0)

>>> math.zeros(x=4).y.unstack(2)
((x=4) float32  0.0, 0.0, 0.0, 0.0, (x=4) float32  0.0, 0.0, 0.0, 0.0)

>>> math.zeros(x=4).x.unstack(2)
AssertionError: Size of dimension x does not match 2.
```

## Non-uniform tensors

The `math` package allows tensors of varying sizes to be stacked into a single tensor.
This tensor then has undefined dimension sizes where the source tensors vary in size.
This is used by `StaggeredGrid`s where the grids holding one vector component have different shapes.

```
>>> math.channel_stack([math.zeros(a=4, b=2), math.zeros(b=2, a=5)], 'c')
(a=None, b=2, c=2) float32
```