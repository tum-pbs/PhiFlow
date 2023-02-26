# Differentiable Geometry

The module [phi.geom](phi/geom) integrates geometric shapes and supports differentiable volumes.

The class [`Geometry`](phi/geom/#phi.geom.Geometry) serves as a base for all geometry objects, such as boxes or spheres.

All properties of `Geometry` support the use of batch, instance and spatial dimensions.
The corresponding values take the type of [Φ-tensors](./Math.html).
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

All built-in basic geometry types support n-dimensional spaces.

[`Spheres`](phi/geom/#phi.geom.Sphere) has two defining properties:

* `.center: Tensor` has a single channel dimension called 'vector'.
* `.radius: Tensor` has no channel dimension


Boxes come in multiple variants:

* [`Box`](phi/geom/#phi.geom.Box) stores the lower and upper corner of the box.
* `Cuboid` stores the center position and half-size.
* [`GridCell`](phi/geom/#phi.geom.GridCell) is similar to `Cuboid` but its spatial dimensions are guaranteed to span a regular grid

[`Points`](phi/geom/#phi.geom.Point) have zero volume and are only characterized by their location.


## Transformations and Operations

Translation: `geometry.shift(delta)`

Rotation: `geometry.rotate(angle)`

Union: `union(geometries)`

Geometries can be inverted using the `~` operator, i.e. the results of 
`lies_inside`, `approximate_signed_distance` and `approximate_fraction_inside` return the inverse values.

Stacking: `GeometryStack(geometries, axis)` allows the type of `Geometry` to vary along a dimension.


## Integration with fields

`Geometry` objects are not [Fields](./Fields.md).
To get a direct `Field` representation from a `Geometry`, use `field.mask()`.
Geometries can be resampled to existing fields using `field.resample()`.
In these cases, the field takes the value `1` inside the geometry and `0` outside.

