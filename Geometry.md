# Differentiable Geometry

The module `phi.geom` \[[source](../phi/geom)\] integrates geometric shapes and supports differentiable volumes.

The base class, `Geometry`, describes the interface that all geometry objects implement.

Most properties of `Geometry` object support the use of batch and spatial dimensions like [tensors](./Math.md).
This allows a single `Geometry` object to describe a collection of shapes with varying properties.

Important properties:

* `.shape: Shape` all batch and spatial dimensions
* `.rank: int` number of spatial dimensions the geometry lives in
* `.center: Tensor` center points of shape (`*geometry.shape`, `vector`)

Important methods:

* `lies_inside(location)` tests if the given points lie inside the geometry
* `approximate_signed_distance(location)` computes the distance of the given points from the surface
* `approximate_fraction_inside(Geometry)` computes the overlap between two geometries.

Geometries can be checked for equality using `==` and `!=`.
They should generally be treated as immutable.


## Basic shapes

Currently, there are two types shapes: boxes and spheres.
Both types can be used in any number of dimensions.

A `Sphere` has two properties:

* `.center: Tensor` has a single channel dimension called 'vector'.
* `.radius: Tensor` has no channel dimension

Both tensors can have any number of batch dimensions.

```
geom.Sphere([0, 0], radius=1)
```

Boxes come in multiple variants:

* `Box` stores the lower and upper corner of the box.
* `Cuboid` stores the center position and half-size.
* `GridCell` is similar to `Cuboid` but its spatial dimensions are guaranteed to span a regular grid

`Box` provides an alternative constructor using Python's indexing syntax.
The slices specify the start and end point of the box where missing values are interpreted as infinity.
```
>>> Box[0:1, :0.5]
Box[1.0xinf at 0.0,-inf]
```

## Transformations and Operations

Translation: `geometry.shift(delta)`

Rotation: `geometry.rotate(angle)`

Union: `union(geometries)`

Geometries can be inverted using the `~` operator, i.e. the results of 
`lies_inside`, `approximate_signed_distance` and `approximate_fraction_inside` return the inverse values.

Stacking: `GeometryStack(geometries, axis)` allows the type of `Geometry` to vary along a dimension.


## Integration with fields

`Geometry` objects are not [Fields](./Fields.md).
However, some sampling operations like `CenteredGrid.sample()` also accept `Geometry` objects.

The classes `SoftGeometryMask` and `HardGeometryMask` represent fields that take the value `1` inside the geometry and `0` outside.
The hard version always returns 0 or 1 while the soft version returns continuous values when volume-sampled.
