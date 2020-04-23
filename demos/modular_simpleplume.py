from phi.flow import *

domain = Domain([80, 64], boundaries=CLOSED)
world.add(StaggeredGrid.sample(0, domain, name='velocity'), physics=IncompressibleVFlow(domain.boundaries))
world.add(CenteredGrid.sample(0, domain, name='marker'), physics=[Drift(), FieldPhysics('marker')])
world.add(FieldEffect(None, targets='velocity'), ProportionalGForce('marker', -0.1))
world.add(Inflow(Sphere(center=(10, 32), radius=5), rate=0.2, target='marker'))

show(App('Modular Simple Plume', framerate=10))
