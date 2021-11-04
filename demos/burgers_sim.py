""" Simulate Burgers' Equation
Simple advection-diffusion equation.
"""
from phi.flow import *


velocity = CenteredGrid(Noise(vector=2), extrapolation.PERIODIC, x=64, y=64, bounds=Box[0:200, 0:100]) * 2

for _ in view(play=False, framerate=10, namespace=globals()).range():
    velocity = diffuse.explicit(velocity, 0.1, dt=1)
    velocity = advect.semi_lagrangian(velocity, velocity, dt=1)
