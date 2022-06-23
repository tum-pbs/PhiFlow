""" Fog Simulation
This simulation models the temperature and humidity of air, producing fog where the humidity is too high.
Cold air streams in from a rectangular window at the top.
"""

from phi.flow import *


DOMAIN = dict(x=64, y=64, bounds=Box(x=100, y=100))
velocity = StaggeredGrid((0, 0), extrapolation.ZERO, **DOMAIN)  # or use CenteredGrid
temperature = CenteredGrid(297.15, extrapolation.BOUNDARY, **DOMAIN)
humidity = 1 * temperature
WINDOW = Box(x=(30, 45), y=(80, 95))
# WINDOW = CenteredGrid(, extrapolation.ZERO, **DOMAIN)
pressure = None
fog = None

for _ in view('fog', temperature, humidity, velocity, 'pressure', play=False, namespace=globals()).range(warmup=1):
    # Window BCs
    temperature = field.where(WINDOW, 283.15, temperature)
    humidity = field.where(WINDOW, 283.15, humidity)
    # Physics
    temperature = diffuse.explicit(advect.mac_cormack(temperature, velocity, dt=1), 0.1, dt=1, substeps=2)
    humidity = advect.mac_cormack(humidity, velocity, dt=1)
    buoyancy_force = temperature * (0, 0.1) @ velocity  # resamples smoke to velocity sample points
    velocity = advect.semi_lagrangian(velocity, velocity, 1) + buoyancy_force
    velocity, pressure = fluid.make_incompressible(velocity, (), Solve('auto', 1e-5, 0, x0=pressure))
    # Compute fog
    fog = field.maximum(humidity - temperature, 0)
