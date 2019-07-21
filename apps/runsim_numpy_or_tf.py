# example that runs a "manual" simple smoke sim either in numpy or TF

from phi.flow import *
from phi.tf.flow import *


DONUMPY    = 1   # main switch, TF (0) or numpy (1)?
res        = 32
dt         = 1.0
steps      = 6   # no of simulation steps
graphSteps = 3   # how many steps to unroll in TF graph


world = World()
if DONUMPY:
    smoke = world.Smoke(Domain([res, res])  ).state  # numpy state
else:
    smoke = world.Smoke(Domain([res, res])  ).state  # numpy state
    #smoke = world.Smoke(Domain([res, res]) , density=placeholder , velocity=placeholder ).state  # TF placeholder state

# NT_DEBUG, todo, "regular" obstacles / inflows dont yet work
#world.Inflow(Sphere((8, 8), radius=4))
#world.Obstacle(box[4:16, 0:8])

print("State type after init: " + format(type( world.state['smoke'][0].density )) )  # NT_DEBUG

domaincache = domain(smoke, () ) # no obstacles
session = Session(Scene.create('data'))
smoke_in  = smoke.copied_with(density=placeholder, velocity=placeholder)

if DONUMPY:
    density = smoke.density
    velocity = smoke.velocity
else:
    density = smoke_in.density
    velocity = smoke_in.velocity

# this example doesnt use dash, write out PNG images via PIL instead
from PIL import Image
def saveImg(a, scale, name):
    ima = np.reshape(a , [a.shape[1],a.shape[2]] ) # remove channel dimension
    im = Image.fromarray( np.asarray(ima * scale, dtype='i') )
    im.save(name) 

# main , step 1: run smoke sim (numpy), or only set up graph for TF

for i in range(steps if DONUMPY else graphSteps):
    # simulation step

    # add manual inflow
    inflow_density = np.zeros([1,res,res,1])  # NT_DEBUG , how to get shape from tensor?
    inflow_density[...,(res//4):(res//2),(res//4):(res//2),0] = 1.
    density = velocity.advect(density, dt=dt) + dt * inflow_density

    #density = velocity.advect(density, dt=dt)  # no inflow
    velocity = stick(velocity, domaincache, dt)
    velocity = velocity.advect(velocity, dt=dt) + dt * buoyancy(density, smoke.gravity, smoke.buoyancy_factor)

    velocity = divergence_free(velocity, domaincache, None, smoke=smoke)

    if i==0:
        print("Density type: "+ format(type(density)) ) # here we either have np array of tf tensor

    if DONUMPY:
        if(i%graphSteps == graphSteps-1): 
            saveImg( density, 10000., "numpy_%04d.png"%i )
        print("Numpy step %d done " % i + ", means " + format(np.mean(density) ) + " " + format(np.mean(velocity.staggered) ) )
    else:
        print("TF graph created for step %d " % i)


# main , step 2: feed to TF (numpy) or do actual run (TF)

if DONUMPY:
    # for numpy run, just feed last computed state to TF, no more simulation to do here...
    smoke_in_data = smoke.copied_with(density=density, velocity=velocity)

    smoke_in = smoke.copied_with(density=placeholder, velocity=placeholder) # dummy output
    smoke_out = smoke_in 

else:
    # for TF, all the work still needs to be done, feed empty state and start simulation
    initDens = np.zeros([1,res,res,1])
    initVel  = np.zeros([1,res+1,res+1,2])
    smoke_in_data = smoke.copied_with(density=initDens, velocity=initVel)

    smoke_out = smoke.copied_with(density=density, velocity=velocity, age=smoke.age + dt)

# run session
for i in range(1 if DONUMPY else (steps//graphSteps)):
    smoke = session.run(smoke_out, {smoke_in: smoke_in_data })

    outDens = np.asarray( smoke.density )
    outVel  = np.asarray( smoke.velocity.staggered )
    if not DONUMPY: 
        # for TF, we only have results now after each graphSteps iterations, write images
        saveImg( outDens, 10000., "tf_%04d.png"%(steps if DONUMPY else graphSteps*(i+1)-1) )
        smoke_in_data = smoke.copied_with(density=outDens, velocity=outVel)

    print("Step session.run %04d"%i + " done, density shape " + format(smoke.density.shape) + ", means " + format(np.mean(outDens) ) + " " + format(np.mean(outVel) ) + ", numpy "+ format(DONUMPY) )
