# Fields

The `phi.physics.field` module provides abstract access to physical quantities in a way that does not depend on the specific data structure used.

The main class, `Field` represents a physical quantity (scalar or vector) that takes a value at any point in space or in a region of space.

All fields are subclasses of `Struct` (see the [documentation](Structs.ipynb)) and have some entries in common:

| Entry             | Type                 | Description                                                                                                                                                                                                                                                                 |
|-------------------|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `component_count` | -                    | Vector length of the Field's values (1 for scalar fields)                                                                                                                                                                                                                   |
| `rank`            | -                    | Dimensions of the physical space (1=1D, 2=2D, 3=3D)                                                                                                                                                                                                                         |
| `data`            | Attribute | Either the actual data as a tensor or a reference to the underlying data structure. In any case, this value supports all element-wise mathematical operators and works with all `phi.math` functions. Some subclasses of `Field` may redefine `data` as a property instead. |
| `name`            | Property             | string                                                                                                                                                                                                                                                                      |
| `flags`           | Property             | Tuple of [`Flag`](../phi/physics/field/flag.py) objects. Flags indicate certain properties about a field such as divergence-freeness and are propagated automatically in mathematical operations.                                                                                   |
| `points`          | -                    | Vector Field holding the sample points. If the Field is not sampled, `points=None`. If the Field is sampled but the sample points vary among the different components of the Field, accessing `Field.points` raises a `StaggeredSamplePoints` exception.                    |


## Build-in Fields

[ConstantField](../phi/physics/field/constant.py): has the same value everywhere.
Has no sample points.

[GeometryMask](../phi/physics/field/mask.py): 1 inside the geometry, 0 outiside.
Has no sample points.

[CenteredGrid](../phi/physics/field/grid.py): has regular sample points.

[StaggeredGrid](../phi/physics/field/staggered_grid.py): has staggered sample points.


## Resampling Fields

Given two fields `field1` and `field2` with different data structures or different sampling points, they can be made compatible using `at` or `sample_at`.

```python
field1.at(field2)  # resamples field1 at the sample points of field 2
```

This assumes that `field2` is actually a sampled field, i.e. that `field2.points` is not `None`.


## Mathematical Operations on Fields

All functions from `phi.math` can be applied to Fields. They act only on `Field.data`.
Operators like +, -, * are also implemented.
Currently, mathematical operations should only be performed on sampled Fields such as CenteredGrid or StaggeredGrid.

Operators working with multiple fields require all fields to be compatible with each other.
Fields without sample points such as ConstantField or GeometryMask are compatible with all Fields.
Other fields can be made compatible using resampling (see above).