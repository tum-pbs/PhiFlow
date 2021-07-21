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
The shift operator `>>` calls the `at()` method on the left field.

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


## Build-in Fields

[`CenteredGrid`](phi/field/#phi.field.CenteredGrid) stores values in a regular grid structure.
The grid values are stored in a `Tensor` whose spatial dimensions match the resolution of the grid.
The `bounds` property stores the physical size of the grid from which the cell size is derived.
`CenteredGrid.elements` is a `GridCell` matching the grid resolution.

[`StaggeredGrid`](phi/field/#phi.field.StaggeredGrid)
stores vector fields in staggered form.
The velocity components are not sampled at the cell centers but at the cell faces.
This results in the `values` having different shapes for the different vector components.
[More on staggered grids](./Staggered_Grids.md).

[`PointCloud`](phi/field/#phi.field.PointCloud)
is a set of points or finite elements, each associated with a value.

[`SoftGeometryMask`](phi/field/#phi.field.SoftGeometryMask) / [`HardGeometryMask`](phi/field/#phi.field.HardGeometryMask):
1 inside the geometry, 0 outside.

[`Noise`](phi/field/#phi.field.Noise)
samples random fluctuations of certain sizes.
Currently, it only supports resampling to grids.

[`AngularVelocity`](phi/field/#phi.field.AngularVelocity)
models a vortex-like velocity field around one or multiple points.
This is useful for sampling the velocity of rotating objects.


## Resampling Fields
Given `val: Field` and `representation: SampledField` with different values structures or different sampling points, 
they can be made compatible using [`at()`](phi/field/#phi.field.Field.at) or `>>`.
```python
val.at(representation)  # resamples val at the elements of representation, keeps extrapolation from val
val >> representation  # like above but the result will have the extrapolation from representation
```
These functions return a `Field` of the same type as `representation`.
If they are already sampled at the same elements, the above operations simply return `val`.
Also, Φ<sub>Flow</sub> may choose optimized code paths for specific combinations, such as two grids with equal sample point spacing `dx`.

Additionally, there are two functions for sampling field values at given locations.

* [`sample`](phi/field/#phi.field.sample) samples the field values at the location of a single geometry or geometry batch.
* [`reduce_sample`](phi/field/#phi.field.reduce_sample) differs from `sample` in that the geometry here describes
  staggered locations at which the individual channel components of the field are stored.
  For centered grids, `sample` and `reduce_sample` are equal.
