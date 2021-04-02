""" Simulate Burgers' Equation
Simple advection-diffusion equation.
"""
from phi.flow import *


DOMAIN = Domain(x=64, y=64, boundaries=PERIODIC, bounds=Box[0:100, 0:100])
velocity = DOMAIN.vector_grid(Noise(vector=2)) * 2

for _ in view(play=False, framerate=10).range():
    velocity = diffuse.explicit(velocity, 0.1, dt=1)
    velocity = advect.semi_lagrangian(velocity, velocity, dt=1)
