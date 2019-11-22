# example that runs a "manual" simple SMOKE sim either in numpy or TF
# note, this example does not use the dash GUI, instead it creates PNG images via PIL

USE_NUMPY = False  # main switch, TF (False) or numpy (True)?

DIM = 2  # 2d / 3d
BATCH_SIZE = 1  # process multiple independent simulations at once
STEPS = 12  # number of simulation STEPS
GRAPH_STEPS = 3  # how many STEPS to unroll in TF graph

RES = 32
DT = 1.0


if USE_NUMPY:
    from phi.flow import *
    import os
else:
    from phi.tf.flow import *


# by default, creates a numpy state, i.e. "SMOKE.density.data" is a numpy array
SMOKE = Smoke(Domain([RES] * DIM, boundaries=OPEN), batch_size=BATCH_SIZE)

if USE_NUMPY:
    DENSITY = SMOKE.density
    VELOCITY = SMOKE.velocity
    # no phiflow session for pure numpy, write to specific directory instead
    IMG_PATH = os.path.expanduser("~/phi/data/manual/numpy")
    os.path.exists(IMG_PATH) or os.mkdir(IMG_PATH) # NT_DEBUG
else:
    SESSION = Session( Scene.create("~/phi/data/manual") )
    IMG_PATH = SESSION._scene.path
    # create TF placeholders with the correct shapes
    SMOKE_IN = SMOKE.copied_with(density=placeholder, velocity=placeholder)
    DENSITY = SMOKE_IN.density
    VELOCITY = SMOKE_IN.velocity


# optional , write images
try:
    from PIL import Image  # for writing PNGs
    
    def save_img(a, scale, name, idx=0):
        if len(a.shape) <= 4:
            ima = np.reshape(a[idx], [a.shape[1], a.shape[2]])  # remove channel dimension , 2d
        else:
            ima = a[idx, :, a.shape[1] // 2, :, 0]  # 3d , middle z slice
        ima = np.reshape(ima, [a.shape[1], a.shape[2]])  # remove channel dimension
        ima = ima[::-1, :]  # flip along y
        im = Image.fromarray(np.asarray(ima * scale, dtype='i'))
        print("    Writing image '"+name+"'")
        im.save(name)

except ImportError:
	def save_img(a, scale, name, idx=0):
		print("    (Skipped writing '"+name+"')")


# main , step 1: run SMOKE sim (numpy), or only set up graph for TF

for i in range(STEPS if USE_NUMPY else GRAPH_STEPS):
    # simulation step; note that the core is only 3 lines for the actual simulation
    # the RESt is setting up the inflow, and debug info afterwards

    INFLOW_DENSITY = math.zeros_like(SMOKE.density)
    if DIM == 2:
        INFLOW_DENSITY.data[..., (RES // 4):(RES // 2), (RES // 4):(RES // 2), 0] = 1.  # (batch, y, x, components)
    else:
        # (batch, z, y, x, components), center along y
        INFLOW_DENSITY.data[..., (RES // 4):(RES // 2), (RES // 4 * 1):(RES // 4 * 3), (RES // 4):(RES // 2), 0] = 1.

    DENSITY = advect.semi_lagrangian(DENSITY, VELOCITY, DT) + DT * INFLOW_DENSITY
    VELOCITY = advect.semi_lagrangian(VELOCITY, VELOCITY, DT) + buoyancy(DENSITY, 9.81, SMOKE.buoyancy_factor) * DT
    VELOCITY = divergence_free(VELOCITY, SMOKE.domain, obstacles=())

    if i == 0:
        print("Density type: %s" % type(DENSITY.data))  # here we either have np array of tf tensor

    if USE_NUMPY:
        if i % GRAPH_STEPS == GRAPH_STEPS - 1:
            save_img(DENSITY.data, 10000., IMG_PATH+"/numpy_%04d.png" % i)
        print("Numpy step %d done, means %s %s" % (i, np.mean(DENSITY.data), np.mean(VELOCITY.staggered_tensor())))
    else:
        print("TF graph created for step %d " % i)


# main , step 2: do actual sim run (TF only)

if not USE_NUMPY:
    # for TF, all the work still needs to be done, feed empty state and start simulation
    SMOKE_OUT = SMOKE.copied_with(density=DENSITY, velocity=VELOCITY, age=SMOKE.age + DT)

    # run session
    for i in range(1 if USE_NUMPY else (STEPS // GRAPH_STEPS)):
        SMOKE = SESSION.run(SMOKE_OUT, feed_dict={SMOKE_IN: SMOKE})  # Passes DENSITY and VELOCITY tensors

        # for TF, we only have RESults now after each GRAPH_STEPS iterations
        save_img(SMOKE.density.data, 10000., IMG_PATH+"/tf_%04d.png" % (GRAPH_STEPS * (i + 1) - 1))

        print("Step SESSION.run %04d done, DENSITY shape %s, means %s %s" %
              (i, SMOKE.density.data.shape, np.mean(SMOKE.density.data), np.mean(SMOKE.velocity.staggered_tensor())))
