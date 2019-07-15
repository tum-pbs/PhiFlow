
# Staggered grids

Staggered grids are a key component of the marker and cell (MAC) method. They sample the velocity components at the centers of the corresponding lower faces of grid cells. This makes them fundamentally different from regular arrays or tensors which are sampled at the cell centers. 
The main advantage of Staggered grids is that it makes computing the divergence straightforward and exact.

![image](./figures/Staggered.png)

In Φ<sub>*Flow*</sub>, staggered grids are represented as instances of [StaggeredGrid](../phi/math/nd.py). They have one more entry in every spatial dimension than corresponding centered fields since the upper face of the upper most cell needs to be included as well.

Staggered grids can either be created directly from an array or tensor holding the staggered values or
using an initializer,

```python
from phi.tf.flow import *
grid = Grid([64, 64])
staggered_zeros = zeros(grid.staggered_shape())
staggered_placeholder = placeholder(grid.staggered_shape(batch_size=16))
```

States such as [Smoke](../phi/physics/smoke.py) that use staggered grids will automatically create one if not provided.

```python
from phi.tf.flow import *
smoke = Smoke(Domain([64, 64]), velocity=placeholder)
```

Staggered grids can also be created from centered fields.

```python
from phi.flow import *; centered_field = zeros([1, 64, 64, 1])

staggered_gradient = StaggeredGrid.gradient(centered_field)
staggered_field_x = StaggeredGrid.from_scalar(centered_field, [1, 0])
```

`StaggeredGrid`s can hold both TensorFlow tensors and NumPy `ndarray`s.
They support basic backend operations and can be passed to `phi.tf.session.Session.run()` like TensorFlow tensors.

Some useful operations include:

```python
from phi.flow import *; smoke = Smoke([64, 64]); velocity = smoke.velocity

# Advect a centered field
advected_density = velocity.advect(smoke.density)
# Advect a staggered field
advected_velocity = velocity.advect(smoke.velocity)

# Compute the curl of a vector potential
curl = velocity.curl()

# Compute the centered divergence field
divergence = velocity.divergence()
```

To get a tensor or ndarray object from a staggered grid, one of the following sampling methods can be used.

```python
from phi.flow import *; smoke = Smoke([64, 64]); velocity = smoke.velocity

staggered_values = velocity.staggered
interpolated_at_centers = velocity.at_centers()
interpolated_at_face_centers = velocity.at_faces(axis=0)
```
