# example that runs a "manual" simple smoke sim either in numpy or TF

from phi.flow import *
from phi.tf.flow import *


useNumpy   = 0   # main switch, TF (0) or numpy (1)?

dim        = 2   # 2d / 3d
bs         = 1   # batch size, process multiple independent simulations at once
steps      = 12  # no of simulation steps
graphSteps = 3   # how many steps to unroll in TF graph

res        = 32
dt         = 1.0

world = World()
# by default, the following call creates a numpy state, i.e. "smoke.density" is a numpy array
smoke = world.Smoke(Domain( np.repeat(res, dim) ) , batch_size=bs).state  
session = Session(Scene.create('data'))
smoke_in  = smoke.copied_with(density=placeholder, velocity=placeholder)

if useNumpy:
    density = smoke.density
    velocity = smoke.velocity
else:
    density = smoke_in.density
    velocity = smoke_in.velocity

# the domaincache initializes flag grid for other ops (like pressure solver in divergence_free() )
# optionally, obstacles can be added here
obstacles = [] # none
#if dim==2:
#    obstacles = [ Obstacle(box[ (res//4*3):(res//4*3+2), (res//4*1):(res//4*3) ]) ] 
#if dim==3:
#    obstacles = [ Obstacle(box[(res//4*3):(res//4*3+2), (res//4*1):(res//4*3), (res//4*1):(res//4*3)]) ] 
domaincache = domain(smoke, obstacles )

# this example does not use dash, instead it creates PNG images via PIL 
from PIL import Image
def saveImg(a, scale, name, idx=0):
    if len( a.shape )<=4:
        ima = np.reshape( a[idx] , [a.shape[1],a.shape[2]] ) # remove channel dimension , 2d
    else:
        ima = a[idx, :, a.shape[1]//2, :,0] # 3d , middle z slice
        ima = np.reshape( ima , [a.shape[1],a.shape[2]] ) # remove channel dimension 
    ima = ima[::-1,:] # flip along y
    im = Image.fromarray( np.asarray(ima * scale, dtype='i') )
    im.save(name) 

# main , step 1: run smoke sim (numpy), or only set up graph for TF

for i in range(steps if useNumpy else graphSteps):
    # simulation step; note that the core is only 4 lines for the actual simulation
    # the rest is setting up the inflow, and debug info afterwards

    # add manual inflow
    id_dim = np.concatenate( ([bs],np.repeat(res, dim),[1]) ) # this simply gives [bs,res,res,1] in 2d
    # note, for TF we could also use: id_dim = density.get_shape().as_list()
    inflow_density = np.zeros( id_dim )
    if dim==2:
        inflow_density[...,(res//4):(res//2), (res//4):(res//2), 0] = 1.
    else:
        inflow_density[...,(res//4):(res//2), (res//4*1):(res//4*3), (res//4):(res//2),0] = 1. # center along y
    density = velocity.advect(density, dt=dt) + dt * inflow_density

    velocity = stick(velocity, domaincache, dt)
    velocity = velocity.advect(velocity, dt=dt) + dt * buoyancy(density, smoke.gravity, smoke.buoyancy_factor)

    velocity = divergence_free(velocity, domaincache, None, smoke=smoke)

    if i==0:
        print("Density type: "+ format(type(density)) ) # here we either have np array of tf tensor

    if useNumpy:
        if(i%graphSteps == graphSteps-1): 
            saveImg( density, 10000., "numpy_%04d.png"%i )
        print("Numpy step %d done " % i + ", means " + format(np.mean(density) ) + " " + format(np.mean(velocity.staggered) ) )
    else:
        print("TF graph created for step %d " % i)


# main , step 2: feed to TF (numpy) or do actual sim run (TF)

if useNumpy:
    # for numpy run, just feed last computed state to TF, no more simulation to do here...
    smoke_in_data = smoke.copied_with(density=density, velocity=velocity)
    smoke_out = smoke_in # output placeholders, could be skipped for numpy version
else:
    # for TF, all the work still needs to be done, feed empty state and start simulation
    initDens = np.zeros( np.concatenate( ([bs], np.repeat(res,   dim),[1])   ))
    initVel  = np.zeros( np.concatenate( ([bs], np.repeat(res+1, dim),[dim]) ))
    smoke_in_data = smoke.copied_with(density=initDens, velocity=initVel) 
    smoke_out = smoke.copied_with(density=density, velocity=velocity, age=smoke.age + dt)

# run session
for i in range(1 if useNumpy else (steps//graphSteps)):
	# version 1: using phiflow states (i.e., smoke states in this case)
    smoke = session.run(smoke_out, {smoke_in: smoke_in_data }) 
    outDens = np.asarray( smoke.density )
    outVel  = np.asarray( smoke.velocity.staggered )

	# version 2 (alternative), using explicit lists 
    #(outDens, outVel) = session.run( (smoke_out.density,smoke_out.velocity), {smoke_in: smoke_in_data })

    if not useNumpy: 
        # for TF, we only have results now after each graphSteps iterations, write images
        saveImg( outDens, 10000., "tf_%04d.png"%(steps if useNumpy else graphSteps*(i+1)-1) )
        smoke_in_data = smoke.copied_with(density=outDens, velocity=outVel)

    print("Step session.run %04d"%i + " done, density shape " + format(smoke.density.shape) + ", means " + format(np.mean(outDens) ) + " " + format(np.mean(outVel) ))
