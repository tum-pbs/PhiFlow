""" Streamline Profile
Simulates a viscous fluid flowing through a horizontal pipe.
"""
from phi.flow import *
from phi.physics._boundaries import STICKY


DOMAIN = Domain(x=50, y=32, boundaries=[OPEN, STICKY])
DT = 1.0
BOUNDARY_MASK = HardGeometryMask(Box[:0.5, :]) >> DOMAIN.staggered_grid()
velocity = DOMAIN.staggered_grid(0)
pressure = DOMAIN.scalar_grid(0)

for _ in view(display='velocity', play=False).range():
    velocity = advect.semi_lagrangian(velocity, velocity, DT)
    velocity = velocity * (1 - BOUNDARY_MASK) + BOUNDARY_MASK * (1, 0)
    velocity, pressure_result = fluid.make_incompressible(velocity, DOMAIN, solve=Solve('CG-adaptive', 1e-5, 0, x0=pressure))
    pressure = pressure_result.x
    velocity = diffuse.explicit(velocity, 0.1, DT)
