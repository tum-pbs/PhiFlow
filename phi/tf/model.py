from __future__ import print_function

from collections import Iterable

from phi.flow import *
from phi.tf.util import *
from phi.tf.session import Session


class TFModel(FieldSequenceModel):

    def __init__(self, name="Network Training", subtitle="Interactive training of a neural network",
                 learning_rate=1e-3,
                 data_validation_fraction=0.6,
                 view_training_data=False,
                 training_batch_size=4,
                 validation_batch_size=16,
                 model_scope_name="model",
                 **kwargs):
        FieldSequenceModel.__init__(self, name=name, subtitle=subtitle, **kwargs)
        self.session = Session(self.scene)
        self.scalars = []
        self.scalar_names = []
        self.editable_placeholders = {}
        self.learning_rate = self.editable_float("Learning_Rate", learning_rate)
        self.training = tf.placeholder(tf.bool, (), "training")
        self.all_optimizers = []
        self.add_trait("tensorflow")
        self.value_view_training_data = view_training_data
        self.training_batch_size = training_batch_size
        self.validation_batch_size = validation_batch_size
        self.train_reader = None
        self.val_reader = None
        self.model_scope_name = model_scope_name
        self.feed_fields = []
        self.shuffle_training_data = True
        self._read_ref = None

    def editable_float(self, name, initial_value, minmax=None, log_scale=None):
        val = EditableFloat(name, initial_value, minmax, None, log_scale)
        setattr(self, "float_"+name.lower(), val)
        placeholder = tf.placeholder(tf.float32, (), name.lower())
        self.add_scalar(name, placeholder)
        self.editable_placeholders[placeholder] = "float_"+name.lower()
        return placeholder

    def editable_int(self, name, initial_value, minmax=None):
        val = EditableInt(name, initial_value, minmax, None)
        setattr(self, "int_"+name.lower(), val)
        placeholder = tf.placeholder(tf.float32, (), name.lower())
        self.add_scalar(name, placeholder)
        self.editable_placeholders[placeholder] = "int_"+name.lower()
        return placeholder

    def prepare(self):
        FieldSequenceModel.prepare(self)

        scalars = [tf.summary.scalar(self.scalar_names[i], self.scalars[i]) for i in range(len(self.scalars))]
        self.merged_scalars = tf.summary.merge(scalars)

        self.info("Initializing variables")
        self.session.initialize_variables()

        model_parameter_count = 0
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_scope_name):
            if not "Adam" in var.name:
                model_parameter_count += int(np.prod(var.get_shape().as_list()))
                # if "conv" in var.name and "kernel" in var.name:
                #     tf.summary.image(var.name, var)
        self.add_custom_property("parameter_count", model_parameter_count)
        self.info("Model variables contain %d total parameters." % model_parameter_count)

        if self.world.batch_size is not None:
            self.training_batch_size = self.world.batch_size
            self.validation_batch_size = self.world.batch_size

        if self.train_reader is not None:
            self.sequence_stride = len(self.train_reader.all_batches(batch_size=self.training_batch_size))
            self.validate()
        else:
            self.info("Preparing model before database is set up.")

        return self

    def minimize(self, name, loss, optimizer=None, reg=None, vars=None):
        assert len(loss.shape) <= 1, "Loss function must be a scalar"
        if not optimizer:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)

        if reg is not None:
            self.add_scalar(name+"_reg", reg)
            optim_function = loss + reg
        else:
            optim_function = loss

        if isinstance(vars, six.string_types):
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vars)

        node = optimizer.minimize(optim_function, var_list=vars)

        self.add_scalar(name, loss)
        self.all_optimizers.append(node)
        return node

    def add_scalar(self, name, node):
        assert isinstance(node, tf.Tensor)
        self.scalar_names.append(name)
        self.scalars.append(node)

    def step(self):
        self.optimize(self.all_optimizers)
        if self.time % self.sequence_stride == 0:
            self.validate(create_checkpoint=True)
        return self

    def optimize(self, optim_nodes, log_loss=False):
        if isinstance(optim_nodes, Iterable):
            optim_nodes = list(optim_nodes)
        else:
            optim_nodes = [optim_nodes]
        scalar_values = self.session.run(optim_nodes + self.scalars, self.feed_dict(self.train_reader, True),
                                     summary_key="train", merged_summary=self.merged_scalars, time=self.time)[1:]
        if log_loss:
            self.info("Optimization: " + ", ".join([self.scalar_names[i]+": "+str(scalar_values[i]) for i in range(len(self.scalars))]))

    def validate(self, create_checkpoint=False):
        # self.info("Running validation...")
        self.session.run(self.scalars, self.feed_dict(self.val_reader, False),
                     summary_key="val", merged_summary=self.merged_scalars, time=self.time)
        if create_checkpoint:
            self.save_model()
        self.info("Validation Done.")

    def base_feed_dict(self):
        return {}

    def feed_dict(self, reader, training, subrange=None):
        base_feed_dict = self.base_feed_dict()
        for placeholder, attrname in self.editable_placeholders.items():
            val = getattr(self, attrname)
            if isinstance(val, EditableValue):
                val = val.initial_value
            base_feed_dict[placeholder] = val
        base_feed_dict[self.training] = training
        if reader is None:
            return base_feed_dict
        else:
            raise NotImplementedError()
            # base_feed_dict[properties] = reader
            # return iterator.fill_feed_dict(base_feed_dict, self.feed_fields, subrange=subrange)

    def val(self, fetches, subrange=None):
        return self.session.run(fetches, self.feed_dict(self.val_iterator, False, subrange=subrange))

    @property
    def view_reader(self):
        return self.train_reader if self.value_view_training_data else self.val_reader

    def view(self, tasks, all_batches=False):
        if tasks is None:
            return None
        if all_batches or self.world.batch_size is not None or isinstance(self.figures.batches, slice):
            return self.session.run(tasks, self.feed_dict(self.view_reader, False))
        else:
            # TODO return Viewable object that indicates batch index
            batches = self.figures.batches
            batch_results = self.session.run(tasks, self.feed_dict(self.view_reader, False, subrange=batches))
            single_task = not isinstance(tasks, (tuple,list))
            if single_task:
                batch_results = [batch_results]
            results = []
            for i, batch_result in enumerate(batch_results):
                is_staggered = isinstance(batch_result, StaggeredGrid)
                if is_staggered: batch_result = batch_result.staggered
                result = [batch_result for i in range(self.validation_batch_size)]
                result = np.concatenate(result)
                results.append(StaggeredGrid(result) if is_staggered else result)
            if single_task:
                return results[0]
            else:
                return results

    def view_batch(self, get_attribute):
        batch = self.view_reader[0:self.validation_batch_size]
        return get_attribute(batch)

    @property
    def read(self):
        return self._read_ref  # TODO set this when val_reader / train_reader are initialized  selector(self.val_reader.struct)

    def save_model(self):
        dir = self.scene.subpath("checkpoint_%08d"%self.time)
        self.session.save(dir)
        return dir

    def load_model(self, checkpoint_dir):
        self.session.restore(checkpoint_dir, scope=self.model_scope_name)

    def model_scope(self):
        return tf.variable_scope(self.model_scope_name)

    def add_field(self, name, field):
        """

        :param name: channel name
        :param field: Tensor, string (database fieldname) or function
        """
        if istensor(field):
            FieldSequenceModel.add_field(self, name, lambda: self.view(field))
        # elif isinstance(field, StructAttributeGetter):
        #     FieldSequenceModel.add_field(self, name, lambda: self.view_batch(field))
        else:
            FieldSequenceModel.add_field(self, name, field)
