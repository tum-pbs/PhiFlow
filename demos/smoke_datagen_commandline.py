from phi.flow import *


world.batch_size = 10  # this multiple simulations are run in parallel
SCENE = Scene.create('~/phi/data/smoke', count=world.batch_size)
print('Created %d scenes starting with %s' % (world.batch_size, SCENE))

FLUID = world.add(Fluid(Domain([64, 64], CLOSED), buoyancy_factor=0.2, batch_size=world.batch_size), physics=IncompressibleFlow())
INFLOW_LOCATIONS = math.stack([[8] * world.batch_size, 8 + math.random_uniform([world.batch_size]) * 48], axis=-1)  # y=8, x=random
world.add(Inflow(Sphere(INFLOW_LOCATIONS, radius=4), rate=0.5))

for i in range(32):
    world.step()
    print("Step %d done , stats: %s %s" % (i, np.mean(FLUID.density.data), np.mean(FLUID.velocity.staggered_tensor())))
    SCENE.write(FLUID.state, frame=i)
    # SCENE.write([FLUID.density, FLUID.velocity], names=["density", "velocity"], frame=i)  # same result

    # another alternative, write all states of the world
    # this gives different filenames, e.g. states_0_density_0000xx.npz (instead of density_0000xx.npz)
    # SCENE.write( world.state, frame=i)

print('Data written to "%s"' % SCENE.path)
