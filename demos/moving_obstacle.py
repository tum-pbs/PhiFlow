from phi.flow import *


def update_obstacle(obstacle, dt, velocity=(0, 1)):
    geometry = obstacle.geometry.shifted(math.as_tensor(velocity) * dt)
    if geometry.center[1] > 70:
        geometry = geometry.shifted([0, -80])
    return obstacle.copied_with(geometry=geometry, velocity=velocity)


smoke = world.add(Fluid(Domain([64, 64], OPEN), buoyancy_factor=0.1), physics=IncompressibleFlow())
world.add(Obstacle(Sphere([32, 20], radius=5)), physics=update_obstacle)
world.step(dt=0.0)

app = App('Moving Objects Demo', dt=0.5, framerate=10)
app.add_field('Velocity', lambda: smoke.velocity)
app.add_field('Domain', lambda: obstacle_mask(world, antialias=True).at(smoke.density))
app.add_field('Divergence', lambda: world.fluid.solve_info['divergence'])
show(app, display=('Domain', 'Velocity'))
