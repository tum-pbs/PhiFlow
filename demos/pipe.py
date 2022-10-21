""" Streamline Profile
Simulates a viscous fluid flowing through a horizontal pipe.
"""
from phi.flow import *

DT = 1.
INFLOW_BC = extrapolation.combine_by_direction(normal=1, tangential=0)
velocity = StaggeredGrid(0, extrapolation.combine_sides(x=(INFLOW_BC, extrapolation.BOUNDARY), y=0), x=50, y=32)
pressure = None

for _ in view('velocity, pressure', namespace=globals()).range():
    velocity = advect.semi_lagrangian(velocity, velocity, DT)
    velocity, pressure = fluid.make_incompressible(velocity, solve=Solve('CG-adaptive', 1e-5, 0, x0=pressure))
    velocity = diffuse.explicit(velocity, 0.1, DT)
