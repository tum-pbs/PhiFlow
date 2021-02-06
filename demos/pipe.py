""" Streamline Profile
Simulates a viscous fluid flowing through a horizontal pipe.
"""
from phi.flow import *
from phi.physics._boundaries import STICKY


DOMAIN = Domain(x=50, y=32, boundaries=[OPEN, STICKY])
DT = 1.0
velocity = DOMAIN.staggered_grid(0)
boundary_mask = HardGeometryMask(Box[:0.5, :]) >> velocity
pressure = DOMAIN.grid(0)

for _ in ModuleViewer(display='velocity').range():
    velocity = advect.semi_lagrangian(velocity, velocity, DT)
    velocity = velocity * (1 - boundary_mask) + boundary_mask * (1, 0)
    velocity, pressure, _iterations, _ = fluid.make_incompressible(velocity, DOMAIN, pressure_guess=pressure)
    velocity = diffuse.explicit(velocity, 0.1, DT)
