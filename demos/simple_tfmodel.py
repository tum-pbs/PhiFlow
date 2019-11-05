from phi.tf.flow import *


resolution = y, x = 64, 64
datapath = '~/phi/data/smoke/' # at least 10 sims, has to match resolution


def network(density):
    """very simple conv net"""
    c = 32  # number of features in inner layers
    W1 = tf.get_variable('W1', [5, 5, 1, c], initializer=tf.random_normal_initializer(stddev=0.01, mean=0.))
    f1 = tf.nn.relu(tf.nn.conv2d(density, W1, strides=[1, 1, 1, 1], padding='SAME'))

    W2 = tf.get_variable('W2', [5, 5, c, c], initializer=tf.random_normal_initializer(stddev=0.01, mean=0.))
    f2 = tf.nn.relu(tf.nn.conv2d(f1, W2, strides=[1, 1, 1, 1], padding='SAME'))

    W3 = tf.get_variable('W3', [3, 3, c, c], initializer=tf.random_normal_initializer(stddev=0.01, mean=0.))
    f3 = tf.nn.relu(tf.nn.conv2d(f2, W3, strides=[1, 1, 1, 1], padding='SAME'))

    Wo = tf.get_variable('Wo', [3, 3, c, 2], initializer=tf.random_normal_initializer(stddev=0.01, mean=0.))
    fo = tf.nn.conv2d(f3, Wo, strides=[1, 1, 1, 1], padding='SAME')  # no activation! we have negative values in the GT fields

    print(format(fo.shape))
    return fo


class TrainingTest(TFApp):

    def __init__(self):
        TFApp.__init__(self, 'Training',
                       learning_rate=2e-4,
                       validation_batch_size=4, training_batch_size=8)
        smoke_in, load_dict = load_state(Smoke(Domain(resolution)))


        with self.model_scope():
            pred_vel = network(smoke_in.density.data)

        target_vel = smoke_in.velocity.staggered_tensor()[..., :-1, :-1, :]
        loss = math.l2_loss(pred_vel - target_vel)
        self.add_objective(loss, 'Loss')

        # this assumes we have 10 sims in the path
        self.set_data(dict=load_dict,
                      train=Dataset.load(datapath, range(0, 8)),
                      val=Dataset.load(datapath, range(8, 10)))

        self.add_field('Velocity (Ground Truth)', smoke_in.velocity)
        self.add_field('Velocity (Model)', pred_vel)
        self.add_field('Density (Input)', smoke_in.density)


show()

# hint, try showing x component only in UI - that one is more interesting than the magnitude
