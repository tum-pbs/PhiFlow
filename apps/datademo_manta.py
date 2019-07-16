from phi.tf.flow import *
import os

dims = 2
data_dir = '/Users/thuerey/Dropbox/shareTmp/temp2/TEST2D_000001/'


class DataDemo(TFModel):

    def __init__(self):
        TFModel.__init__(self, 'Data Demo')

        # this is the original resolution of the mantaflow sim
        # allocate one size smaller so that velocity matches, and crop
        # scalar fields to (mantaflowRes-1) via MantaScalar() channels
        mantaflowRes = 48

        if dims==2:
            smoke = world.Smoke(Domain([mantaflowRes-1, mantaflowRes-1])) # 2D YXc
        if dims==3:
            smoke = world.Smoke(Domain([mantaflowRes-1, mantaflowRes-1, mantaflowRes-1]))  # 3D ZYXc
                    
        smoke.velocity = smoke.density = placeholder # switch to TF tensors
        state_in = smoke.state
        state_out = world.step(smoke) # generates tensors now
                
        # set up manta reader 
        reader = BatchReader( Dataset.load(data_dir) , (MantaScalar('pressure'), 'vel') ) 

        i = 0 
        for batch in reader.all_batches(batch_size=1):
            #print("Shape batch mantaload "+format(batch[0].shape)+" "+ format(batch[1].shape))  
            state = smoke.copied_with(density=batch[0], velocity=batch[1])

            self.session.run(state_out, {state_in: state}) # give to TF
            i = i + 1
        print("MantaScalar demo done, %d batches read " % i)

        exit(1)

app = DataDemo().show(framerate=1, production=__name__!="__main__") 
