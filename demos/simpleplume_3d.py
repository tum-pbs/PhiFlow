from phi.flow import *

world.add(Fluid(Domain([40, 32, 32], boundaries=CLOSED), buoyancy_factor=0.1), physics=IncompressibleFlow(pressure_solver=SparseCG()))
world.add(Inflow(Sphere(center=(5, 16, 16), radius=3), rate=0.2))

show(App('Simple Plume 3D', framerate=10))
