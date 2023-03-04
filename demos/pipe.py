""" Streamline Profile
Simulates a viscous fluid flowing through a horizontal pipe.
"""
from phi.flow import *

DT = 1.
velocity = StaggeredGrid(0, extrapolation.combine_sides(x=(vec(x=1, y=0), ZERO_GRADIENT), y=0), x=50, y=32)
pressure = None

for _ in view('velocity, pressure', namespace=globals()).range():
    velocity = advect.semi_lagrangian(velocity, velocity, DT)
    velocity = diffuse.explicit(velocity, 0.1, DT)
    velocity, pressure = fluid.make_incompressible(velocity, solve=Solve('CG-adaptive', 1e-5, x0=pressure))
