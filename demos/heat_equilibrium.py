""" Heat Relaxation

A horizontal plate at the top is heated and a sphere at the bottom is cooled.
Control the heat source using the sliders at the bottom.
"""
from phi.flow import *


DOMAIN = dict(x=64, y=64, extrapolation=0)
DT = 1.0
x = control(32, (14, 50))
y = control(20, (4, 40))
radius = control(4, (2, 10))
temperature = CenteredGrid(0, **DOMAIN)

for _ in view(temperature, framerate=30, namespace=globals()).range():
    temperature -= DT * CenteredGrid(Box(x=None, y=(44, 46)), **DOMAIN)
    temperature += DT * CenteredGrid(Sphere(x=x, y=y, radius=radius), **DOMAIN)
    temperature = diffuse.explicit(temperature, 0.5, DT, substeps=4)
