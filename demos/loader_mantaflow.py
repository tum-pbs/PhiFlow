# this example tries to load "pressure_XXXXXX.npz" and "vel_XXXXXX.npz" files
# from all simulations sim_XXXXXX in the given scene path

import sys

# instead of full phi.flow or phi.tf.flow, import specific modules only
from phi.tf.flow import *

SCENE_PATH = sys.argv[1] if len(sys.argv) >= 2 else '~/phi/data/simpleplume'
SCENE_PATH = os.path.expanduser(SCENE_PATH)

# this is the original resolution of the mantaflow sim
# allocate one size smaller so that velocity matches, and crop
# scalar fields to (MANTAFLOW_RESOLUTION-1) via MantaScalar() channels
MANTAFLOW_RESOLUTION = 64

# 2D or 3D
DIMS = 2


class DataLoader(App):

    def __init__(self, scene_path, dims, mantaflowRes):
        App.__init__(self, 'Data Demo')

        smoke = world.add(Fluid(Domain([mantaflowRes - 1] * dims)), physics=IncompressibleFlow())  # 2D: YXc , 3D: ZYXc
        smoke.velocity = smoke.density = placeholder  # switch to TF tensors
        state_in = smoke.state
        state_out = world.step(smoke)  # generates tensors now

        # set up manta reader
        reader = BatchReader(Dataset.load(scene_path), (MantaScalar('pressure'), 'vel'))

        idx = 0
        m_list = []
        for idx, batch in enumerate(reader.all_batches(batch_size=1)):
            # batch[0] is a numpy array with pressure now, batch[1] has the velocities
            # could be modified here; warning - note uses pressure to density here...
            state = smoke.copied_with(density=batch[0], velocity=batch[1])
            # collect some statistics
            m_list.append([np.mean(np.abs(state.density.data)), np.mean(np.abs(state.velocity.staggered_tensor()))])
            self.session.run(state_out, {state_in: state})  # pass to TF
            # self.session.run(state_out, {state_in: (batch[0], batch[1])}) # alternative, without state copy
            # now we have the tensor version in state_out

        print("MantaScalar demo done, %d batches read, abs-mean %f " % (idx, np.mean(np.asarray(m_list))))


# note, no GUI, use viewer.py instead to display
APP = DataLoader(scene_path=SCENE_PATH, dims=DIMS, mantaflowRes=MANTAFLOW_RESOLUTION)
