from phi.flow import *
physics_config.x_first()

world.add(Fluid(Domain([50, 32], boundaries=[OPEN, STICKY]), buoyancy_factor=0.1), physics=[IncompressibleFlow(), lambda fluid, dt: fluid.copied_with(velocity=diffuse(fluid.velocity, 0.1 * dt))])
world.add(ConstantVelocity(box[:1, :], velocity=(1, 0)))

app = App('Streamline Profile', 'Vertical Pipe')
app.add_field('Velocity', lambda: world.fluid.velocity.padded(1))
show(app)
