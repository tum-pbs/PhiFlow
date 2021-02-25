""" Simulate Burgers' Equation
Simple advection-diffusion equation.
"""
from phi.flow import *


DOMAIN = Domain(x=64, y=64, boundaries=PERIODIC, bounds=Box[0:100, 0:100])
velocity = Noise(vector=2) * 2 >> DOMAIN.vector_grid()

for _ in ModuleViewer(framerate=10).range():
    velocity = diffuse.explicit(velocity, 0.1, dt=1)
    velocity = advect.semi_lagrangian(velocity, velocity, dt=1)
