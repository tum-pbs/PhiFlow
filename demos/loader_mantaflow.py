from phi.tf.flow import *
import os

# this example tries to load "pressure_XXXXXX.npz" and "vel_XXXXXX.npz" files
# from the simulations sim_XXXXXX in the given directory
scene_path = sys.argv[1] if len(sys.argv) >= 2 else '~/phi/data/simpleplume'
scene_path = os.path.expanduser(scene_path)

# this is the original resolution of the mantaflow sim
# allocate one size smaller so that velocity matches, and crop
# scalar fields to (mantaflowRes-1) via MantaScalar() channels
mantaflowRes = 64

# 2D or 3D
dims = 2

class DataLoader(TFModel):

    def __init__(self):
        TFModel.__init__(self, 'Data Demo')

        smoke = world.Smoke(Domain([mantaflowRes-1] * dims)) # 2D: YXc , 3D: ZYXc
        smoke.velocity = smoke.density = placeholder # switch to TF tensors
        state_in = smoke.state
        state_out = world.step(smoke) # generates tensors now
                
        # set up manta reader 
        reader = BatchReader( Dataset.load(scene_path) , (MantaScalar('pressure'), 'vel') ) 

        i = 0 
        for batch in reader.all_batches(batch_size=1):

            # batch[0] is a numpy array with pressure now, batch[1] has the velocities
            # could be modified here; warning - note uses pressure to density here...
            state = smoke.copied_with(density=batch[0], velocity=batch[1])

            self.session.run(state_out, {state_in: state}) # give to TF
            #self.session.run(state_out, {state_in: (batch[0], batch[1]) }) # alternative, without state copy
            
            # now we have tensor version in state_out
            i = i + 1

        print("MantaScalar demo done, %d batches read " % i)

        exit(1)

# note, no GUI , use viewer.py instead to display
app = DataLoader()
