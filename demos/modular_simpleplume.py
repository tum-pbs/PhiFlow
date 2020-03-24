from phi.flow import *


class ModularSimpleplume(App):

    def __init__(self, domain=Domain([80, 64], boundaries=CLOSED)):
        App.__init__(self, framerate=10)
        world.add(StaggeredGrid.sample(0, domain, name='velocity'), physics=IncompressibleVFlow(domain.boundaries))
        world.add(CenteredGrid.sample(0, domain, name='marker'), physics=[Drift(), FieldPhysics('marker')])
        world.add(FieldEffect(None, targets='velocity'), ProportionalGForce('marker', -0.1))
        world.add(Inflow(Sphere(center=(10, 32), radius=5), rate=0.2, target='marker'))


show(ModularSimpleplume())
