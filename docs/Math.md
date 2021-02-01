# Math

The [`phi.math` module](phi/math) provides abstract access to tensor operations.
It internally uses NumPy/SciPy, TensorFlow or PyTorch to execute the actual operations, depending on which backend is selected (see below).
This ensures that code written against `phi.math` functions produces equal results on all backends.

To that end, `phi.math` provides a new `Tensor` class which should be used instead of directly accessing native tensors from NumPy, TensorFlow or PyTorch. 
While similar to the native tensor classes, `phi.math.Tensor`s have named and typed dimensions.

When performing operations such as `+, -, *, /, %, **` or calling `math` functions on `Tensor`s, dimensions are matched by name and type.
This eliminates the need for manual reshaping or the use of singleton dimensions.

Example:
```python
from phi import math

math.ones(x=10) + math.ones(x=10)
# Out: (x=10) float32  2.0 < ... < 2.0

math.ones(x=5) + math.ones(batch=10)
# Out: (batch=10, x=5) float32  2.0 < ... < 2.0
```


## Shapes

The shape of a `Tensor` is represented by a `Shape` object which can be accessed as `tensor.shape`.
In addition to the integer sizes of the dimensions, the shape also stores the names of the dimensions as strings as well as their types.

There are three types of dimensions

* **Batch** dimensions are ignored by most operations. They are automatically added as needed.
* **Spatial** dimensions are associated with physical space. If two `Tensors`s live in different physical spaces, operations may raise `IncompatibleShapes` errors.
* **Channel** dimensions typically list vector elements or feature maps. They are automatically added as needed.

The preferred way to define a `Shape` is via the `shape()` function.
It takes the dimension sizes as keyword arguments.

```python
math.shape(batch=10, y=2, x=4, vector=2)
# Out: (batch=10, y=2, x=4, vector=2)
```

The dimension types are inferred from the names according to the following rules:

* Single letter &rarr; Spatial dimension
* Starts with 'vector' &rarr; Channel dimension
* Else &rarr; Batch dimension

```python
math.shape(batch=10, y=2, x=4, vector=2).types
# Out: ('batch', 'spatial', 'spatial', 'channel')
```

`Shape` objects should be considered *immutable*.
Do not change any property of a `Shape` directly.

Important `Shape` properties (see the [API documentation](phi/math/#phi.math.Shape) for a full list):

* `.sizes: tuple` enumerates the sizes as ints or None, similar to NumPy's shapes
* `.names: tuple` enumerates the dimension names
* `.rank: int = len(shape)` number of dimensions
* `.named_sizes` to iterate over `name, size` of each dimension
* `.batch: Shape` / `.spatial: Shape` / `.channel: Shape` contains only batch / spatial / channel dimensions
* `.non_batch: Shape` / `.non_spatial: Shape` / `.non_channel: Shape` contains only the other two types of dimensions
* `.batch_rank: Shape` / `.spatial_rank: Shape` / `.channel_rank: Shape` alias for `.batch.rank`
* `.volume` number of elements a tensor of this shape contains

Important `Shape` methods:

* `get_size(name)` returns the size of a dimension
* `index(name)` returns the position of a dimension
* `&` (`combined(Shape)`) returns a shape containing the dimensions from both shapes
* `expand_<type>(size, name, pos)` adds a dimension to the shape
* `extend(Shape)` adds all dimensions from the other shape to this one
* `without(dims)` drops the specified dimensions
* `only(dims)`, `select(*names)` drops all other dimensions

Additional tips and tricks

* `tuple(shape)` is equal to `shape.sizes`.
* `'x' in shape` tests whether a dimension by the name of 'x' is present.
* `shape1 == shape2` tests equality including names, types and order of dimensions.
* `shape1 + shape2` adds the sizes of the shapes.
* `shape1 & shape2` combines the shapes.
* `shape.x` returns the size of the dimension 'x'.
* `spatial_shape(Shape)` can be used to get the spatial part or create a new `Shape` depending on the input.


## Tensor Creation

The `tensor()` function converts a scalar, a `list`, a `tuple`, a NumPy array or a TensorFlow/PyTorch tensor to a `Tensor`.
The dimension names can be specified using the `names` keyword and dimension types are inferred from the names.
Otherwise, they are determined automatically.

```python
math.tensor([1, 2, 3])
# Out: (1, 2, 3) along vector

math.tensor(numpy.zeros([1, 5, 1, 2]), names='batch, x,y, vector')
# Out: (batch=1, x=5, y=1, vector=2) float64  0.0 < ... < 0.0

math.tensor(numpy.zeros([3, 3, 1]), names=['y', 'x', 'time'])
# Out: (y=3, x=3, time=1) float64  0.0 < ... < 0.0
```

There are a couple of functions in the `phi.math` module for creating basic tensors.

* `zeros()`
* `ones()`
* `random_normal()`
* `random_uniform()`
* `meshgrid()`

Most functions allow the shape of the tensor to be specified via a `Shape` object or alternatively through the keyword arguments.
In the latter case, the dimension types are inferred from the names.
```python
math.zeros(x=5, y=4)  # Tensor with two spatial dimensions
# Out: (x=5, y=4) float32  0.0 < ... < 0.0

math.zeros(math.shape(y=4, x=5))
# Out: (y=4, x=5) float32  0.0 < ... < 0.0

math.meshgrid(x=5, y=(0, 1, 2))
# Out: (x=5, y=3, vector=2) int64  0 < ... < 4
```


## Backend Selection

The `phi.math` library does not implement basic operators directly but rather delegates the calls to another computing library.
Currently, it supports three such libraries: NumPy/SciPy, TensorFlow and PyTorch.
These are referred to as *backends*.

The easiest way to use a certain backend is via the import statement:

* `phi.flow` &rarr; NumPy/SciPy
* `phi.tf.flow` &rarr; TensorFlow
* `phi.torch.flow` &rarr; PyTorch

This determines what backend is used to create new tensors.
Existing tensors created with a different backend will keep using that backend.
For example, even if TensorFlow is set as the default backend, NumPy-backed tensors will continue using NumPy functions.

The global backend can be set directly using `math.backend.set_global_default_backend()`.
Backends also support context scopes, i.e. tensors created within a `with backend:` block will use that backend to back the new tensors.
The three backends can be referenced via the global variables `phi.math.SCIPY_BACKEND`, `phi.tf.TF_BACKEND` and `phi.torch.TORCH_BACKEND`.

When passing tensors of different backends to one function, an automatic conversion will be performed,
e.g. NumPy arrays will be converted to TensorFlow or PyTorch tensors.



## Indexing, Slicing, Unstacking

The recommended way of indexing or slicing tensors is using the syntax
```python
tensor.<dim>[start:end:step]
```
This can be chained to index multiple dimensions.
```python
tensor.x[:2].y[1:-1]
```

Alternatively tensors can be indexed using a dictionary of the form `tensor[{dim: slice or int}]`.

Tensors can be unstacked along any dimension using `t.unstack(dim)` or `t.<dim>.unstack()`.
When passing the dimension size to the latter, tensors can even be unstacked along dimensions they do not posess.

```python
math.zeros(x=4).x.unstack()
# Out: (0.0, 0.0, 0.0, 0.0)

math.zeros(x=4).y.unstack(2)
# Out: ((0.0, 0.0, 0.0, 0.0) along x, (0.0, 0.0, 0.0, 0.0) along x)

math.zeros(x=4).x.unstack(2)
# Out: AssertionError: Size of dimension x does not match 2.
```

## Non-uniform Tensors

The `math` package allows tensors of varying sizes to be stacked into a single tensor.
This tensor then has dimension sizes of type `Tensor` where the source tensors vary in size.

One use case of this are `StaggeredGrid`s where the tensors holding the vector components have different shapes.
```python
math.channel_stack([math.zeros(a=4, b=2), math.zeros(b=2, a=5)], 'c')
# Out: (a=(4, 5) along c, b=2, c=2) float32  0.0 < ... < 0.0
```

Non-uniform tensors have the property that their second-order shape has more than one dimension.
```python
math.channel_stack([math.zeros(a=4, b=2), math.zeros(b=2, a=5)], 'c').shape.shape
# Out: (dims=3, c=2)
```



## Data Types and Precision

The package `phi.math` provides a custom `DataType` class that can be used with all backends.
There are no global variables for common data types; instead you can create one by specifying the kind and length in bits.
```python
float32 = math.DType(float, 32)
int64 = math.DType(int, 64)
complex128 = math.DType(complex, 128)
bool_ = math.DType(bool)

float32
# Out: float32

float32.kind
# Out: <class 'float'>

int64.itemsize
# Out: 8

complex128.bits
# Out: 128

complex128.precision
# Out: 64
```

By default, floating point operations use 32 bit (single precision).
This can be changed globally using `math.set_global_precision(64)` or locally using `with math.precision(64):`.

This setting does not affect integers.
To specify the number of integer bits, use `math.to_int()` or cast the data type directly using `math.cast()`.
