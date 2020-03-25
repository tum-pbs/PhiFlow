from phi.flow import *

domain = Domain([64, 64], boundaries=PERIODIC)
world.add(BurgersVelocity(domain, velocity=lambda s: math.randfreq(s) * 2), physics=Burgers())

show(App('Burgers Equation in %dD' % len(domain.resolution), framerate=5))
