""" Taylor-Green Vortex
"""
from phi.flow import *


def taylor_green_pressure(x):
    return math.sum(math.cos(2 * x * Q), 'vector') / 4 * math.exp(-4 * Q**2 * t / RE)


def taylor_green_velocity(x):
    sin = math.sin(Q*x)
    cos = math.cos(Q*x)
    return math.exp(-2*Q**2*t/RE) * math.stack({
        'x': -cos.vector['x'] * sin.vector['y'],
        'y': sin.vector['x'] * cos.vector['y']},
        dim=channel('vector'))


DOMAIN = dict(x=64, y=64, bounds=Box(x=2*math.pi, y=2*math.pi), extrapolation=extrapolation.PERIODIC)
RE = vis.control(1.)
Q = vis.control(1.)
dt = vis.control(0.01)
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
