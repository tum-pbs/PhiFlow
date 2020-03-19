# Staggered grids

Staggered grids are a key component of the marker-and-cell (MAC) method. They sample the velocity components at the centers of the corresponding lower faces of grid cells. This makes them fundamentally different from regular arrays or tensors which are sampled at the cell centers.
A central advantage of Staggered grids is that it makes operations such as computing the divergence of a flow field straightforward and exact.

![image](./figures/Staggered.png)

In Φ<sub>*Flow*</sub>, staggered grids are represented as instances of [StaggeredGrid](../phi/physics/field/staggered_grid.py) and implement the [Field API](Fields.md).
Since each voxel has two faces per dimension, staggered grids contain more values than the corresponding centered grids.
In memory, each component of a staggered grid is held in a different array while on disk, a single array, called `staggered_tensor`, is stored.

When using a built-in simulation such as Fluid, staggered grids are generated automatically from the provided values.
New grids can also be created from the simulation object.
```python
from phi.tf.flow import *

centered_zeros = fluid.centered_grid('f0', 0)
centered_zeros = CenteredGrid.sample(0, fluid.domain)
staggered_zeros = fluid.staggered_grid('v', 0)
```


Staggered grids can be created manually from an array or tensor holding the staggered values or using an initializer,

```python
from phi.tf.flow import *

velocity_tensor = np.zeros([1, 65, 65, 2])
staggered_field = StaggeredGrid(velocity_tensor)
```

States such as [fluid](../phi/physics/fluid.py) ([documentation](Fluid_Simulation.md)) that use staggered grids will automatically create one if not provided.

```python
from phi.tf.flow import *

fluid = Fluid(Domain([64, 64]), velocity=placeholder)
```

Staggered grids can also be created from centered fields. The first example below stores the spatial
derivatives for each axis in the staggered grid, while the second call initializes a staggered grid with
the size of the centered one, and multiples it with (2,1) for x and y components, respectively, to obtain
a 2D vector quantity. Note that because the centered grid is interpolated at the faces of the staggered one,
by default the values at the boundary will drop off:

```python
from phi.flow import *

centered_field = CenteredGrid(np.ones([1, 64, 64, 1]), 1)

staggered_gradient = StaggeredGrid.gradient(centered_field)
staggered_field = StaggeredGrid.sample(centered_field * [1, 2], domain)
```

`StaggeredGrid`s can hold both TensorFlow `Tensor`'s and NumPy `ndarray`s.
They support basic backend operations and can be passed to `phi.tf.session.Session.run()` like TensorFlow tensors.

Some useful operations include:

To get a `Tensor` or `ndarray` object from a staggered grid, one of the following sampling methods can be used.

```python
staggered_values = velocity.staggered_tensor()
interpolated_at_centers = velocity.at_centers()  # or velocity.at(velocity.center_points)
interpolated_at_x_face_centers = velocity.at(velocity.data[-1])
```
