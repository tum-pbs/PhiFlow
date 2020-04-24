from phi.flow import *

world.add(Fluid(Domain([80, 64], boundaries=CLOSED), buoyancy_factor=0.1), physics=IncompressibleFlow())
world.add(Inflow(Sphere(center=(10, 32), radius=5), rate=0.2))

show(App('Simple Plume', framerate=10))
