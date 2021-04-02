""" Heat Relaxation

A horizontal plate at the top is heated and a sphere at the bottom is cooled.
Control the heat source using the sliders at the bottom.
"""
from phi.flow import *


DOMAIN = Domain(x=64, y=64)
DT = 1.0
x = control(32, (14, 50))
y = control(20, (4, 40))
radius = control(4, (2, 10))
temperature = DOMAIN.scalar_grid(0)

for _ in view(temperature, framerate=30).range():
    temperature -= DT * DOMAIN.scalar_grid(Box[0:64, 44:46])
    temperature += DT * DOMAIN.scalar_grid(Sphere([x, y], radius=radius))
    temperature = diffuse.explicit(temperature, 0.5, DT, substeps=4)
