from phi.flow import *

path = '~/phi/simpleplume'
scene = Scene.create(path)

smoke = world.Smoke(Domain([32, 32], SLIPPERY))
smoke.buoyancy_factor = 0.2 
px = 8 + int(np.random.rand() * 16)
world.Inflow(Sphere((8, px), 4), rate=0.5) 

for i in range(30):
    world.step() 
    print("Step %d done " % i + ", stats: " + format(np.mean(smoke.density) ) + " " + format(np.mean(smoke.velocity.staggered) ) )

    # write smoke sim state
    smoke = world.state.states[0]
    scene.write( smoke, frame=i) 

    # alternatively, specify which fields to write with:
    #scene.write([smoke.density, smoke.velocity, smoke._last_pressure], names=["density","velocity_staggered","pressure"], frame=i)

    # another alternative, write all states of the world:
    #scene.write( world.state, frame=i) 

print("Data written to '"+path+"'")
