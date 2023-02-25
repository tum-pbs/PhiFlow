# Fields

The [`phi.field`](phi/field/) module contains various data structures - such as grids or point clouds - 
and provides a common interface to access them.
This allows the physics to be independent of the underlying data structure to some degree.


## Abstract classes

The [`Field`](phi/field/#phi.field.Field) class is the base class that all fields extend.
It represents a physical quantity `F(x)` that defines a value at every point `x` in n-dimensional space.
The values of `F(x)` may have any number of dimensions, described by the channel dimensions of the Field.
Scalar fields have no channel dimensions, vector fields have one, etc.

Important properties:

* `.shape: Shape` contains batch and spatial dimensions from 
* `.spatial_rank: int = len(shape.spatial)` is the dimensionality of physical space

Important methods

* `sample_at(Tensor) -> Tensor` computes the field values at the given points
* `sample_in(Geometry) -> Tensor` computes the field values in the given volumes
* `at(SampledField) -> SampledField` returns a field with the same sample points as the specified representation.
* `unstack(dim) -> tuple[Field]` slices the field along a dimension

Fields implement many mathematical operators, e.g. `+, -, * , /, **`.
The shift operator `@` calls the `at()` method on the left field.

The class [`SampledField`](phi/field/#phi.field.SampledField) extends `Field` to form the basis for all fields that explicitly store their data.
The most important sampled fields are 
[`CenteredGrid`](phi/field/#phi.field.CenteredGrid), 
[`StaggeredGrid`](phi/field/#phi.field.StaggeredGrid) and 
[`PointCloud`](phi/field/#phi.field.PointCloud).

Important properties:

* `.values: Tensor` data that is used in sampling
* `.elements: Geometry` sample points as finite volumes
* `.points: Tensor` center points of `elements`
* `.extrapolation: Extrapolation` determines how values outside the region covered by `values` are determined.

Non-sampled fields inherit from `AnalyticField`.
They model `F(x)` as a function instead of from data.


## Built-in Fields

[`CenteredGrid`](phi/field/#phi.field.CenteredGrid) stores values in a regular grid structure.
The grid values are stored in a `Tensor` whose spatial dimensions match the resolution of the grid.
The `bounds` property stores the physical size of the grid from which the cell size is derived.
`CenteredGrid.elements` is a `GridCell` matching the grid resolution.

[`StaggeredGrid`](phi/field/#phi.field.StaggeredGrid)
stores vector fields in staggered form.
The velocity components are not sampled at the cell centers but at the cell faces.
This results in the `values` having different shapes for the different vector components.
[More on staggered grids](Staggered_Grids.html).

[`PointCloud`](phi/field/#phi.field.PointCloud)
is a set of points or finite elements, each associated with a value.

[`Noise`](phi/field/#phi.field.Noise)
samples random fluctuations of certain sizes.
Currently, it only supports resampling to grids.

[`AngularVelocity`](phi/field/#phi.field.AngularVelocity)
models a vortex-like velocity field around one or multiple points.
This is useful for sampling the velocity of rotating objects.


## Resampling Fields
Given `val: Field` and `representation: SampledField` with different values structures or different sampling points, 
they can be made compatible using [`at()`](phi/field/#phi.field.Field.at) or `@`.
```python
val.at(representation, keep_extrapolation=False)  # resamples val at the elements of representation
val @ representation  # same as above
```
These functions return a `Field` of the same type as `representation`.
If they are already sampled at the same elements, the above operations simply return `val`.
Φ<sub>Flow</sub> may choose optimized code paths for specific combinations, such as two grids with equal sample point spacing `dx`.

When resampling staggered grids with `keep_extrapolation=True`, the sample points of the resampled field may be different from `representation`.
This is because the sample points and value tensor shape of staggered grids depends on the extrapolation type.

Additionally, there are two functions for sampling field values at given locations.

* [`sample`](phi/field/#phi.field.sample) samples the field values at the location of a single geometry or geometry batch.
* [`reduce_sample`](phi/field/#phi.field.reduce_sample) differs from `sample` in that the geometry here describes
  staggered locations at which the individual channel components of the field are stored.
  For centered grids, `sample` and `reduce_sample` are equal.


## Extrapolations

Sampled fields, such as [`CenteredGrid`](phi/field/#phi.field.CenteredGrid),
[`StaggeredGrid`](phi/field/#phi.field.StaggeredGrid)
[`PointCloud`](phi/field/#phi.field.PointCloud) all have an `extrapolation` member variable of type 
[`Extrapolation`](phi/math/extrapolation.html#phi.math.extrapolation.Extrapolation).
The extrapolation determines the values outside the region in which the field is sampled.
It takes the place of the boundary condition (e.g. Neumann / Dirichlet) which would be used in a mathematical formulation.

### Extrapolations vs Boundary Conditions

While both extrapolation and traditional boundary conditions fill the same role, there are a couple of differences between the two.
Boundary conditions determine the field values (or a spatial derivative thereof) at the boundary of a volume, i.e. they cover an n-1 dimensional region.
Extrapolations, on the other hand, cover everything outside the sampled volume, i.e. an n-dimensional region.

Numerical methods working directly with traditional boundary conditions have to treat the boundaries separately (e.g. different stencils).
With extrapolations, the same computations can typically be achieved by first padding the field and then applying a single operation everywhere.
This makes low-order methods more efficient, especially on GPUs or TPUs where fewer kernels need to be launched, reducing the overhead.
Also, user code typically is more concise and expressive with extrapolations.

### Standard Extrapolations

Standard extrapolation types are listed [here](phi/math/extrapolation.html#header-variables).

* `PERIODIC` copies the values from the opposite side.
* `BOUNDARY` copies the closest value from the grid. For the boundary condition *∂u/∂x = 0*, this is accurate to second order.
* `ConstantExtrapolation`, such as `ZERO` or `ONE` fill the outside with a constant value.
  For a boundary condition *u=c*, the first padded value is exact and values padded further out are accurate to first order.

Custom extrapolations can be implemented by extending the
[`Extrapolation`](phi/math/extrapolation.html#phi.math.extrapolation.Extrapolation) class.
Extrapolations also support a limited set of arithmetic operations, e.g. `PERIODIC * ZERO = ZERO`.

### Specifying Extrapolations per Side

Different extrapolation types can be chosen for each side of a domain, e.g. a closed box with an open top.
This can be achieved using [`combine_sides()`](phi/math/extrapolation.html#phi.math.extrapolation.combine_sides)
which allows the extrapolations to be specified by dimension.

The following example uses 0 for the upper face along `y` and 1 everywhere else.
```python
zero_top = extrapolation.combine_sides(x=extrapolation.ONE, y=(extrapolation.ONE, extrapolation.ZERO))
```
For a full example, see the [pipe demo](https://github.com/tum-pbs/PhiFlow/blob/master/demos/pipe.py).