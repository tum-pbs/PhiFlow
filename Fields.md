# Fields

The `phi.field` module \[[source](../phi/field/)\] contains various data structures - such as grids or point clouds - 
and provides a common interface to access them.
This allows the physics to be independent of the underlying data structure to some degree.


## Abstract classes

The `Field` class \[[source](../phi/field/_field.py)\] is the base class that all fields extend.
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

The class `SampledField` \[[source](../phi/field/_field.py)\] extends `Field` to form the basis for all fields that explicitly store their data.
The most important sampled fields are `CenteredGrid`, `StaggeredGrid` and `PointCloud`.

Important properties:

* `.values: Tensor` data that is used in sampling
* `.elements: Geometry` sample points as finite volumes
* `.points: Tensor` center points of `elements`
* `.extrapolation: Extrapolation` determines how values outside the region covered by `values` are determined.

Non-sampled fields inherit from `AnalyticField`.
They model `F(x)` as a function instead of from data.


## Build-in Fields

`ConstantField` models `F(x) = const.`
\[[source](../phi/field/_constant.py)\]

`CenteredGrid` \[[source](../phi/field/_grid.py)\] stores values in a regular grid structure.
The grid values are stored in a `Tensor` whose spatial dimensions match the resolution of the grid.
The `bounds` property stores the physical size of the grid from which the cell size is derived.
`CenteredGrid.elements` is a `GridCell` matching the grid resolution.

`StaggeredGrid` \[[source](../phi/field/_grid.py)\]
stores vector fields in staggered form.
The velocity components are not sampled at the cell centers but at the cell faces.
This results in the `values` having different shapes for the different vector components.
[More on staggered grids](./Staggered_Grids.md).

`PointCloud` \[[source](../phi/field/_point_cloud.py)\]
is a set of points or finite elements, each associated with a value.

`GeometryMask` \[[source](../phi/field/_mask.py)\]:
1 inside the geometry, 0 outside.

`Noise` \[[source](../phi/field/_noise.py)\]
samples random fluctuations of certain sizes.
Currently, it only supports resampling to grids.

`AngularVelocity` \[[source](../phi/field/_angular_velocity.py)\]
models a vortex-like velocity field around one or multiple points.
This is useful for sampling the velocity of rotating objects.


## Resampling Fields

Given `field1: Field` and `field2: SampledField` with different values structures or different sampling points, they can be made compatible using `at()` or `>>`.

```python
field1.at(field2)  # resamples field1 at the elements of field 2
field1 >> field2  # same operation
```

If they are already sampled at the same elements, the above operations simply return `field1`.

Note that `at()` is based on the volume sampling method `sample_in()`.
To sample at the center points, use `field1.sample_at(field2.points)`.
