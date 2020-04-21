from phi.flow import *


def obstacle_at(time):
    return Sphere([32, (time + 20) % 64], radius=5)


smoke = world.add(Fluid(Domain([64, 64], CLOSED), buoyancy_factor=0.1), physics=IncompressibleFlow())
world.add(Obstacle(obstacle_at(0)), physics=GeometryMovement(obstacle_at))
world.step(dt=0.0)

app = App('Moving Objects Demo', dt=0.5, framerate=10)
app.add_field('Velocity', lambda: smoke.velocity)
app.add_field('Domain', lambda: obstacle_mask(world, antialias=True).at(smoke.density))
app.add_field('Divergence', lambda: world.fluid.solve_info['divergence'])
show(app, display=('Domain', 'Velocity'))
