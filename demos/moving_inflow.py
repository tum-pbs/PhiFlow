from phi.flow import *


def inflow_at(time):
    return Sphere([10, 32 + 15 * math.sin(time * 0.1)], radius=5)


smoke = world.add(Fluid(Domain([64, 64], CLOSED), buoyancy_factor=0.1), physics=IncompressibleFlow())
world.add(Inflow(inflow_at(0), rate=0.2), physics=GeometryMovement(inflow_at))

show(App('Moving Objects Demo', dt=0.5, framerate=10))
