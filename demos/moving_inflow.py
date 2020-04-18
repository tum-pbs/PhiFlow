from phi.flow import *

# Note that while obstacles can technically move, they don't affect the simulation correctly. This will be implemented later on.


def obstacle_at(time):
    return Sphere([32, (time + 20) % 64], radius=5)


def inflow_at(time):
    return Sphere([10, 32 + 15 * math.sin(time * 0.1)], radius=5)


smoke = world.add(Fluid(Domain([64, 64], CLOSED), buoyancy_factor=0.1), physics=IncompressibleFlow())
world.add(Inflow(inflow_at(0), rate=0.2), physics=GeometryMovement(inflow_at))
world.add(Obstacle(obstacle_at(0)), physics=GeometryMovement(obstacle_at))

app = App('Moving Objects Demo', dt=0.5, framerate=10)
app.add_field('Density', lambda: smoke.density)
app.add_field('Velocity', lambda: smoke.velocity)
app.add_field('Domain', lambda: obstacle_mask(smoke).at(smoke.density))
show(app, display=('Density', 'Domain'))
