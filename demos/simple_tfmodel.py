from phi.tf.flow import *


RESOLUTION = y, x = 64, 64
DATAPATH = '~/phi/data/smoke/' # at least 10 sims, has to match RESOLUTION


def network(density):
    """very simple conv net"""
    n_feat = 32  # number of features in inner layers
    w_1 = tf.get_variable('w_1', [5, 5, 1, n_feat], initializer=tf.random_normal_initializer(stddev=0.01, mean=0.))
    f_1 = tf.nn.relu(tf.nn.conv2d(density, w_1, strides=[1, 1, 1, 1], padding='SAME'))

    w_2 = tf.get_variable('w_2', [5, 5, n_feat, n_feat], initializer=tf.random_normal_initializer(stddev=0.01, mean=0.))
    f_2 = tf.nn.relu(tf.nn.conv2d(f_1, w_2, strides=[1, 1, 1, 1], padding='SAME'))

    w_3 = tf.get_variable('w_3', [3, 3, n_feat, n_feat], initializer=tf.random_normal_initializer(stddev=0.01, mean=0.))
    f_3 = tf.nn.relu(tf.nn.conv2d(f_2, w_3, strides=[1, 1, 1, 1], padding='SAME'))

    Wo = tf.get_variable('Wo', [3, 3, n_feat, 2], initializer=tf.random_normal_initializer(stddev=0.01, mean=0.))
    fo = tf.nn.conv2d(f_3, Wo, strides=[1, 1, 1, 1], padding='SAME')  # no activation! we have negative values in the GT fields

    print(format(fo.shape))
    return fo


class TrainingTest(TFApp):

    def __init__(self):
        TFApp.__init__(self, 'Training',
                       learning_rate=2e-4,
                       validation_batch_size=4, training_batch_size=8)
        smoke_in, load_dict = load_state(Smoke(Domain(RESOLUTION)))


        with self.model_scope():
            pred_vel = network(smoke_in.density.data)

        target_vel = smoke_in.velocity.staggered_tensor()[..., :-1, :-1, :]
        loss = math.l2_loss(pred_vel - target_vel)
        self.add_objective(loss, 'Loss')

        # this assumes we have 10 sims in the path
        self.set_data(dict=load_dict,
                      train=Dataset.load(DATAPATH, range(0, 8)),
                      val=Dataset.load(DATAPATH, range(8, 10)))

        self.add_field('Velocity (Ground Truth)', smoke_in.velocity)
        self.add_field('Velocity (Model)', pred_vel)
        self.add_field('Density (Input)', smoke_in.density)


show()

# hint, try showing x component only in UI - that one is more interesting than the magnitude
