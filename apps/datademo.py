from phi.tf.flow import *
import os


data_dir = ''
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)
    # TODO write some data


class DataDemo(TFModel):

    def __init__(self):
        TFModel.__init__(self, 'Data Demo')

        smoke = Smoke(Domain([64, 64]))
        state_in = placeholder(smoke.shape())
        state_out = smoke.step(state_in)

        reader = BatchReader(Dataset.load(data_dir), SmokeState('Density', 'Velocity'))
        for batch in reader.all_batches(batch_size=5):
            self.session.run(state_out, {state_in: batch})

        iterator = reader.all_batches(batch_size=5, last=WRAP, loop=True)
        iterator.batch_size = 3
        next(iterator)

        viewbatch = reader[0:16]

        # for batch in datait.iterate(batch_size=1):
        #     batch_result = self.session.run(state_out, {state_in: batch})