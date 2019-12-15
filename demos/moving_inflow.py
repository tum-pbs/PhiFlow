from phi.flow import *

# Note that while obstacles can technically move, they don't affect the simulation correctly. This will be implemented later on.


def obstacle_at(time):
    return Sphere([32, (time + 20) % 64], radius=5)


def inflow_at(time):
    return Sphere([10, 32 + 15 * math.sin(time * 0.1)], radius=5)


class MovingInflowDemo(App):

    def __init__(self):
        App.__init__(self, 'Moving Objects Demo', framerate=10)
        smoke = world.add(Fluid(Domain([64, 64], CLOSED), buoyancy_factor=0.1), physics=IncompressibleFlow())
        world.add(Inflow(inflow_at(0), rate=0.2), physics=GeometryMovement(inflow_at))
        world.add(Obstacle(obstacle_at(0)), physics=GeometryMovement(obstacle_at))
        self.add_field('Density', lambda: smoke.density)
        self.add_field('Velocity', lambda: smoke.velocity)
        self.add_field('Domain', lambda: obstacle_mask(smoke).at(smoke.density))


show(display=('Density', 'Domain'))
