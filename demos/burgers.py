""" Simulate Burgers' Equation
Simple advection-diffusion equation.
"""
from phi.flow import *


velocity = CenteredGrid(Noise(vector=2), PERIODIC, x=64, y=64, bounds=Box(x=200, y=100)) * 2


# @jit_compile  # for PyTorch, TensorFlow and Jax
def burgers_step(v, dt=1.):
    v = diffuse.explicit(v, 0.1, dt=dt)
    v = advect.semi_lagrangian(v, v, dt=dt)
    return v


for _ in view(play=False, framerate=10, namespace=globals()).range():
    velocity = burgers_step(velocity)
