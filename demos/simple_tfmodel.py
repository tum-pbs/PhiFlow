from phi.tf.flow import *
import tensorflow as tf

simSize  = 64 # sim size
datapath = '~/phi/data/datagen/' # data to load, has to match sim size, at least 10 sims

# setup very simple conv net 
def network(density):
    c = 32 # number of features in inner layers
    W1 = tf.get_variable("W1", [5,5, 1,c], initializer=tf.random_normal_initializer(stddev=0.01, mean=0.))
    f1 = tf.nn.relu( tf.nn.conv2d(density, W1, strides=[1,1,1,1], padding="SAME") )

    W2 = tf.get_variable("W2", [5,5, c,c], initializer=tf.random_normal_initializer(stddev=0.01, mean=0.))
    f2 = tf.nn.relu( tf.nn.conv2d(f1, W2, strides=[1,1,1,1], padding="SAME") )

    W3 = tf.get_variable("W3", [3,3, c,c], initializer=tf.random_normal_initializer(stddev=0.01, mean=0.))
    f3 = tf.nn.relu( tf.nn.conv2d(f2, W3, strides=[1,1,1,1], padding="SAME") )

    Wo = tf.get_variable("Wo", [3,3, c,2], initializer=tf.random_normal_initializer(stddev=0.01, mean=0.))
    fo = tf.nn.conv2d(f3, Wo, strides=[1,1,1,1], padding="SAME") # no activation! we have negative values in the GT fields
    print(format(fo.shape))
    return fo

def crop(b): 
    return b[...,0:b.shape[1]-1, 0:b.shape[2]-1,:] # crop 1 layer

class TrainingTest(TFModel):

    def __init__(self):
        TFModel.__init__(self, "Training", 
            learning_rate=2e-4,
            validation_batch_size=4, training_batch_size=8)
        smoke = world.Smoke(Domain([simSize] * 2), density=placeholder, velocity=placeholder)

        with self.model_scope():
            pred_vel = network(smoke.density)

        target_vel = crop(smoke.velocity.staggered)
        loss = l2_loss(pred_vel - target_vel)
        self.add_objective(loss, "Loss")
        
        # this assumes we have 10 sims in the path
        self.set_data( 
            train=Dataset.load( datapath,  range(0, 8)),
            val=  Dataset.load( datapath,  range(8,10)), 
            placeholders=smoke.state )

        self.add_field("Velocity (Ground Truth)", smoke.velocity)
        self.add_field("Velocity (Model)", pred_vel)
        #self.add_field("Density (Input)", smoke.density) # optionally show density input

# hint, try showing x component only in UI - that one is more interesting than the magnitude
app = TrainingTest().show(production=__name__!="__main__")
