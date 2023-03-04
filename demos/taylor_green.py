""" Taylor-Green Vortex
"""
from phi.flow import *


def taylor_green_pressure(x):
    return math.sum(math.cos(2 * x * VORTEX_COUNT), 'vector') / 4 * math.exp(-4 * VORTEX_COUNT ** 2 * t / RE)


def taylor_green_velocity(x):
    sin = math.sin(VORTEX_COUNT * x)
    cos = math.cos(VORTEX_COUNT * x)
    return math.exp(-2 * VORTEX_COUNT ** 2 * t / RE) * stack({
        'x': -cos.vector['x'] * sin.vector['y'],
        'y': sin.vector['x'] * cos.vector['y']},
        dim=channel('vector'))


DOMAIN = dict(x=64, y=64, bounds=Box(x=2*PI, y=2*PI), extrapolation=PERIODIC)
VORTEX_COUNT = 1
RE = vis.control(60.)  # Reynolds number for analytic function
dt = vis.control(0.1)
t = 0.

analytic_pressure = sim_pressure = CenteredGrid(taylor_green_pressure, **DOMAIN)
analytic_velocity = sim_velocity = StaggeredGrid(taylor_green_velocity, **DOMAIN)  # also works with CenteredGrid

viewer = view(sim_velocity, sim_pressure, namespace=globals())
for _ in viewer.range():
    t += dt
    sim_velocity = advect.semi_lagrangian(sim_velocity, sim_velocity, dt)
    sim_velocity, sim_pressure = fluid.make_incompressible(sim_velocity)
    analytic_pressure = CenteredGrid(taylor_green_pressure, **DOMAIN)
    analytic_velocity = StaggeredGrid(taylor_green_velocity, **DOMAIN)
