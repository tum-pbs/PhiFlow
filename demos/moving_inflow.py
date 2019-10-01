from phi.flow import *
import math


def obstacle_at(time):
    return Sphere([32, (time + 20) % 64], radius=5)


def inflow_at(time):
    return Sphere([10, 32 + 15*math.sin(time * 0.1)], radius=5)


class MovingInflowDemo(FieldSequenceModel):

    def __init__(self):
        FieldSequenceModel.__init__(self, 'Moving Objects Demo', stride=5)
        smoke = world.Smoke(Domain([64, 64], SLIPPERY))
        inflow = world.Inflow(inflow_at(0), rate=0.2)
        inflow.physics = GeometryMovement(inflow_at)
        obstacle = Obstacle(obstacle_at(0), SLIPPERY)
        obstacle.physics = GeometryMovement(obstacle_at)
        self.add_field('Density', lambda: smoke.density)
        self.add_field('Velocity', lambda: smoke.velocity)
        self.add_field('Domain', lambda: smoke.domaincache.active(1))


app = MovingInflowDemo().show(display=('Density', 'Velocity'), production=__name__ != '__main__')
