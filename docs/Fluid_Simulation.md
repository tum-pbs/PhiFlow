# Writing Fluid Simulations in Φ<sub>Flow</sub>

There are two main viewpoints for simulating fluids:

* *Eulerian* simulations use grids, tracking fluid distribution and fluid velocity at fixed sample points
* *Lagrangian* simulations track particles that move with the fluid.

Φ<sub>Flow</sub> supports both methods to some extent but mainly focuses on Eulerian simulations.


## Operators

Fluid simulations typically employ operator splitting to break up the various terms in the equation.
The most common ones - advection, projection and diffusion - are explained in the following.

### Advection (Transport)

Advection models the movement of the fluid.
For Lagrangian simulations, this means moving the particle locations without affecting other properties.
In Eulerian simulations, all the grid values have to be adjusted to account for the flow.

The corresponding functions are located in `phi.physics.advect` which is part of the standard import and take `Field` instances as arguments.

For Eulerian simulations, the velocity may be sampled at the cell centers (`CenteredGrid`) or in [staggered form](Staggered_Grids.md) at the face centers (`StaggeredGrid`).
The following example demonstrates semi-Lagrangian advection \[Stam 1999\] with both types of velocities.
```python
from phi.flow import *

DOMAIN = Domain(x=64, y=96)
for grid_type in (StaggeredGrid, CenteredGrid):
    velocity = DOMAIN.vgrid(Noise(vector=2), type=grid_type)
    temperature = DOMAIN.grid(Noise())
    
    advected_temperature = advect.semi_lagrangian(temperature, velocity, dt=1)
    advected_velocity = advect.semi_lagrangian(velocity, velocity, dt=1)  # self-advection
```

For Lagrangian simulations, the quantities of interest are represented as `PointCloud` instances.
To advect the points of a `PointCloud`, the function `advect.points(field, velocity, dt)` may be used.

If the velocity is sampled on a grid but the advected quantity is a `PointCloud`, `advect.runge_kutta_4(field, velocity, dt)` is a good option.


### Projection (Incompressibility)

The projection operation may be used to make a given velocity grid divergence-free \[Chorin and Temam 1968\].
This is a simple way of ensuring the fluid is incompressible as advection on a divergence-free field is volume-preserving.

This operation works best with [staggered grids](Staggered_Grids.md) as they allow for an exact computation of the divergence.

The function `fluid.make_incompressible(velocity, domain, obstacles, solve_params, pressure_guess)` solves a linear system of equations to compute the pressure.
It then subtracts the pressure gradient from the velocity to obtain a divergence-free field.
In addition to the divergence-free velocity, it returns the computed pressure, the number of solve iterations and the initial divergence.

This operator has no equivalent for Lagrangian simulations.
Instead, velocities may be resampled to a grid to apply the projection step and then resampled back to the particle locations.
The methods FLIP and PIC make use of variations of this technique.

The following snippet demonstrates how to resample a `PointCloud` velocity to a grid, be it centered or staggered.
```python
from phi.flow import *

DOMAIN = Domain(x=64, y=80)
velocity = PointCloud(Sphere(positions, 1), values)
velocity_grid = velocity.sample_in(DOMAIN.cells)
velocity_grid = velocity >> DOMAIN.grid()
velocity_grid = velocity >> DOMAIN.staggered_grid()
```
Here, the `>>` operator is a shorthand for calling `velocity.at(...)` and `grid()` without arguments creates a grid with all values being zero.

Sampling back to particles works the same way, i.e. `velocity_grid >> velocity`.


### Diffusion (Viscosity)

Diffusion may be applied as a separate step or added to the projection step by altering the system of linear equations.

For explicit diffusion on grids, you can use the function `field.diffuse(field, diffusivity, dt, substeps=1)`.
Note that for large amounts of diffusion (i.e. `diffusivity * dt > cell size`), the `substeps` argument must be increased for the result to remain stable.



## Examples

### Single-phase flow

The following example runs a Eulerian single-phase flow simulation.
This is suited for settings such as air flow or a filled tank of water.

```python
from phi.flow import *

DOMAIN = Domain(x=64, y=80, boundaries=CLOSED, bounds=Box[0:100, 0:100])
velocity = DOMAIN.staggered_grid(Noise())
pressure = DOMAIN.grid(0)
for _ in range(100):
    velocity = advect.semi_lagrangian(velocity, velocity, dt=1)
    velocity, pressure, iterations, _ = fluid.make_incompressible(velocity, DOMAIN, pressure_guess=pressure)
```
