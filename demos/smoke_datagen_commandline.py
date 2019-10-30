from phi.flow import *


world.batch_size = 1  # this many simulations are run in parallel
scene = Scene.create('~/phi/data/smoke', count=world.batch_size)

smoke = world.add(Smoke(Domain([64, 64], SLIPPERY), buoyancy_factor=0.2))
world.add(Inflow(Sphere((8, 8 + int(np.random.rand() * 16)), radius=4), rate=0.5))

for i in range(30):
    world.step()
    print("Step %d done , stats: %s %s" % (i, np.mean(smoke.density.data), np.mean(smoke.velocity.staggered_tensor())))
    scene.write(smoke.state, frame=i)

    # alternatively, specify which fields to write with:
    #scene.write([smoke.density, smoke.velocity, smoke._last_pressure], names=["density","velocity_staggered","pressure"], frame=i)

    # another alternative, write all states of the world 
    # this gives different filenames, e.g. states_0_density_0000xx.npz (instead of density_0000xx.npz)
    #scene.write( world.state, frame=i) 

print('Data written to "%s"' % scene.path)
