# example that runs a "manual" simple smoke sim either in numpy or TF


use_numpy = True  # main switch, TF (False) or numpy (True)?

dim = 2  # 2d / 3d
batch_size = 1  # process multiple independent simulations at once
steps = 12  # number of simulation steps
graph_steps = 3  # how many steps to unroll in TF graph

res = 32
dt = 1.0


if use_numpy:
    from phi.flow import *
else:
    from phi.tf.flow import *


# by default, creates a numpy state, i.e. "smoke.density.data" is a numpy array
smoke = Smoke(Domain([res] * dim, boundaries=OPEN), batch_size=batch_size)

if use_numpy:
    density = smoke.density
    velocity = smoke.velocity
else:
    session = Session(Scene.create('data'))
    # create TF placeholders with the correct shapes
    smoke_in = smoke.copied_with(density=placeholder, velocity=placeholder)
    density = smoke_in.density
    velocity = smoke_in.velocity


from PIL import Image  # this example does not use the dash GUI, instead it creates PNG images via PIL
import os
os.path.exists('images') or os.mkdir('images')


def save_img(a, scale, name, idx=0):
    if len(a.shape) <= 4:
        ima = np.reshape(a[idx], [a.shape[1], a.shape[2]])  # remove channel dimension , 2d
    else:
        ima = a[idx, :, a.shape[1] // 2, :, 0]  # 3d , middle z slice
        ima = np.reshape(ima, [a.shape[1], a.shape[2]])  # remove channel dimension
    ima = ima[::-1, :]  # flip along y
    im = Image.fromarray(np.asarray(ima * scale, dtype='i'))
    im.save(name)


# main , step 1: run smoke sim (numpy), or only set up graph for TF

for i in range(steps if use_numpy else graph_steps):
    # simulation step; note that the core is only 3 lines for the actual simulation
    # the rest is setting up the inflow, and debug info afterwards

    inflow_density = math.zeros_like(smoke.density)
    if dim == 2:
        inflow_density.data[..., (res // 4):(res // 2), (res // 4):(res // 2), 0] = 1.  # (batch, y, x, components)
    else:
        # (batch, z, y, x, components), center along y
        inflow_density.data[..., (res // 4):(res // 2), (res // 4 * 1):(res // 4 * 3), (res // 4):(res // 2),0] = 1.

    density = advect.semi_lagrangian(density, velocity, dt) + dt * inflow_density
    velocity = advect.semi_lagrangian(velocity, velocity, dt) + buoyancy(density, 9.81, smoke.buoyancy_factor) * dt
    velocity = divergence_free(velocity, smoke.domain, obstacle_mask=None)

    if i == 0:
        print("Density type: %s"% type(density.data))  # here we either have np array of tf tensor

    if use_numpy:
        if (i % graph_steps == graph_steps - 1):
            save_img(density.data, 10000., "images/numpy_%04d.png" % i)
        print("Numpy step %d done, means %s %s" % (i, np.mean(density.data), np.mean(velocity.staggered_tensor())))
    else:
        print("TF graph created for step %d " % i)


# main , step 2: do actual sim run (TF only)

if not use_numpy:
    # for TF, all the work still needs to be done, feed empty state and start simulation
    smoke_out = smoke.copied_with(density=density, velocity=velocity, age=smoke.age + dt)

    # run session
    for i in range(1 if use_numpy else (steps // graph_steps)):
        smoke = session.run(smoke_out, feed_dict={smoke_in: smoke})  # Passes density and velocity tensors

        # for TF, we only have results now after each graph_steps iterations
        save_img(smoke.density.data, 10000., "images/tf_%04d.png" % (graph_steps * (i + 1) - 1))

        print("Step session.run %04d done, density shape %s, means %s %s" %
              (i, smoke.density.data.shape, np.mean(smoke.density.data), np.mean(smoke.velocity.staggered_tensor())))

    session._scene.remove()
    os.rmdir('data')