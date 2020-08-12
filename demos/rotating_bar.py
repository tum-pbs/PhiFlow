from phi.flow import *
physics_config.x_first()


def update_obstacle(obstacle, dt, angular_velocity=0.1):
    geometry = obstacle.geometry.rotated(angular_velocity * dt)
    return obstacle.copied_with(geometry=geometry, angular_velocity=angular_velocity)


world.add(Fluid(Domain([128, 128], CLOSED, box=box([100, 100]))), physics=IncompressibleFlow())
world.add(Obstacle(box[10:90, 48:52], name='bar'), physics=update_obstacle)
world.step()

app = App('Moving Objects Demo', framerate=10)
app.add_field('Velocity', lambda: world.fluid.velocity)
app.add_field('Domain', lambda: mask(world.bar.geometry, antialias=True).at(world.fluid.density))
app.add_field('Divergence', lambda: world.fluid.solve_info['divergence'])
show(app, display=('Domain', 'Velocity'))
