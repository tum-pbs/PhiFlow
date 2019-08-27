from phi.flow import *

path = '~/phi/simpleplume'
scene = Scene.create(path)

smoke = world.Smoke(Domain([32, 32], SLIPPERY))
smoke.buoyancy_factor = 0.2 
px = 8 + int(np.random.rand() * 16)
world.Inflow(Sphere((8, px), 4), rate=0.5) 

for i in range(30):
    world.step() 
    smoke = world.state.states[0]
    scene.write( smoke, frame=i) 
    print("Step %d done " % i + ", stats: " + format(np.mean(smoke.density) ) + " " + format(np.mean(smoke.velocity.staggered) ) )

print("Data written to '"+path+"'")
