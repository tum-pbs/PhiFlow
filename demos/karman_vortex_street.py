"""Karman Vortex Street
Air flow around a static cylinder.
Vortices start appearing after a couple of hundred steps.
"""
from phi.flow import *  # minimal dependencies
# from phi.torch.flow import *
# from phi.tf.flow import *
# from phi.jax.flow import *


SPEED = vis.control(2.)
velocity = StaggeredGrid((SPEED, 0), extrapolation.BOUNDARY, x=128, y=128, bounds=Box(x=128, y=64))
CYLINDER = Obstacle(geom.infinite_cylinder(x=15, y=32, radius=5, inf_dim=None))
BOUNDARY_MASK = StaggeredGrid(Box(x=(-INF, 0.5), y=None), velocity.extrapolation, velocity.bounds, velocity.resolution)
pressure = None


@jit_compile  # Only for PyTorch, TensorFlow and Jax
def step(v, p, dt=1.):
    v = advect.semi_lagrangian(v, v, dt)
    v = v * (1 - BOUNDARY_MASK) + BOUNDARY_MASK * (SPEED, 0)
    return fluid.make_incompressible(v, [CYLINDER], Solve('auto', 1e-5, 0, x0=p))


for _ in view('vorticity,velocity,pressure', namespace=globals()).range():
    velocity, pressure = step(velocity, pressure)
    vorticity = field.curl(velocity)
