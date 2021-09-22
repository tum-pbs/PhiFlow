""" Streamline Profile
Simulates a viscous fluid flowing through a horizontal pipe.
"""
from phi.flow import *

DOMAIN = dict(x=50, y=32, extrapolation=extrapolation.combine_sides(x=extrapolation.BOUNDARY, y=extrapolation.ZERO))
DT = 1.0
BOUNDARY_MASK = StaggeredGrid(HardGeometryMask(Box[:0.5, :]), **DOMAIN)
velocity = StaggeredGrid(0, **DOMAIN)
pressure = None

for _ in view('velocity, pressure', namespace=globals()).range():
    velocity = advect.semi_lagrangian(velocity, velocity, DT)
    velocity = velocity * (1 - BOUNDARY_MASK) + BOUNDARY_MASK * (1, 0)
    velocity, pressure = fluid.make_incompressible(velocity, solve=Solve('CG-adaptive', 1e-5, 0, x0=pressure))
    velocity = diffuse.explicit(velocity, 0.1, DT)
