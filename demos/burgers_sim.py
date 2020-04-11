from phi.flow import *

domain = Domain([64, 64], boundaries=PERIODIC, box=box[0:100, 0:100])
world.add(BurgersVelocity(domain, velocity=Noise(channels=domain.rank) * 2), physics=Burgers())

show(App('Burgers Equation in %dD' % len(domain.resolution), framerate=5))
