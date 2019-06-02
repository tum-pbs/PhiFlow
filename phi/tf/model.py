from __future__ import print_function
from phi.model import *
from phi.tf.flow import *
from phi.data import *
from phi.tf.util import istensor


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
        self.summary_directory = self.scene.subpath("summary")
        self.profiling_directory = self.scene.subpath("profile")
        self.scalars = []
        self.scalar_names = []
        self.editable_placeholders = {}
        self.learning_rate = self.editable_float("Learning_Rate", learning_rate)
        self.training = tf.placeholder(tf.bool, (), "training")
        self.recent_optimizer_node = None
        self.add_trait("tensorflow")
        self.database = Database(Dataset("train", 1-data_validation_fraction, (DATAFLAG_TRAIN,)),
                                 Dataset("val", data_validation_fraction, (DATAFLAG_TEST,)))
        self.value_view_training_data = view_training_data
        self.training_batch_size = training_batch_size
        self.validation_batch_size = validation_batch_size
        self.train_iterator = None
        self.model_scope_name = model_scope_name

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
        if self.train_iterator is not None:
            self.info("Loading initial data...")
            self.feed_dict(self.train_iterator, True)  # is this still necessary?
            self.feed_dict(self.val_iterator, False)
            self.feed_dict(self.view_train_iterator, False)
            self.info("Done.")
            self.sequence_stride = self.train_iterator.batch_count
            if not np.isfinite(self.sequence_stride): self.sequence_stride = 1
            self.validate()
        else:
            self.info("Preparing model before database is set up.")
        return self

    def step(self):
        self.tfstep()
        return self

    def finalize_setup(self, feed_fields, log_batch_retrieval=False, shuffle_training_data=True):
        assert self.sim is not None, "TFModel.sim must be set before finalize_setup"
        self.sim.summary_directory = self.summary_directory

        scalars = [tf.summary.scalar(self.scalar_names[i], self.scalars[i]) for i in range(len(self.scalars))]
        self.merged_scalars = tf.summary.merge(scalars)

        self.sim.initialize_variables()

        self.feed_fields = feed_fields

        model_parameter_count = 0
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_scope_name):
            if not "Adam" in var.name:
                model_parameter_count += int(np.prod(var.get_shape().as_list()))
                # if "conv" in var.name and "kernel" in var.name:
                #     tf.summary.image(var.name, var)
        self.add_custom_property("parameter_count", model_parameter_count)

        if self.sim.batch_size:
            self.training_batch_size = self.sim.batch_size
            self.validation_batch_size = self.sim.batch_size
        logf = self.info if log_batch_retrieval else None
        if self.database.scene_count > 0:
            self.train_iterator = self.database.linear_iterator("train", feed_fields, self.training_batch_size, shuffled=shuffle_training_data, logf=logf)
            self.val_iterator = self.database.fixed_range("val", feed_fields, range(self.validation_batch_size))
            self.view_train_iterator = self.database.fixed_range("train", feed_fields, range(self.validation_batch_size))
        else:
            self.train_iterator = None
            self.val_iterator = None
            self.view_train_iterator = None

    def minimizer(self, name, loss, optimizer=None, reg=None, vars=None):
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
        self.recent_optimizer_node = node
        return node

    def add_scalar(self, name, node):
        assert isinstance(node, tf.Tensor)
        self.scalar_names.append(name)
        self.scalars.append(node)

    def tfstep(self, optimizer=None):
        if not optimizer:
            optimizer = self.recent_optimizer_node
        self.optimize(optimizer)
        if self.time % self.sequence_stride == 0:
            self.validate(create_checkpoint=True)

    def optimize(self, optim_node):
        scalar_values = self.sim.run([optim_node] + self.scalars, self.feed_dict(self.train_iterator, True),
                                     summary_key="train", merged_summary=self.merged_scalars, time=self.time)[1:]
        # self.info("Optimization Done.") #+", ".join([self.scalar_names[i]+": "+str(scalar_values[i]) for i in range(len(self.scalars))]))

    def validate(self, create_checkpoint=False):
        # self.info("Running validation...")
        self.sim.run(self.scalars, self.feed_dict(self.val_iterator, False),
                     summary_key="val", merged_summary=self.merged_scalars, time=self.time)
        if create_checkpoint:
            self.save_model()
        self.info("Validation Done.")

    def base_feed_dict(self):
        return {}

    def feed_dict(self, iterator, training, subrange=None):
        base_feed_dict = self.base_feed_dict()
        for placeholder, attrname in self.editable_placeholders.items():
            val = getattr(self, attrname)
            if isinstance(val, EditableValue):
                val = val.initial_value
            base_feed_dict[placeholder] = val
        base_feed_dict[self.training] = training
        if iterator is None:
            return base_feed_dict
        else:
            return iterator.fill_feed_dict(base_feed_dict, self.feed_fields, subrange=subrange)

    def val_dict(self, subrange=None):
        iterator = self.view_train_iterator if self.value_view_training_data else self.val_iterator
        return self.feed_dict(iterator, False, subrange=subrange)

    def view(self, tasks, options=None, run_metadata=None, all_batches=False):
        if tasks is None:
            return None
        if all_batches or self.sim.batch_size is not None or isinstance(self.figures.batches, slice):
            return self.sim.run(tasks, self.val_dict(), options=options, run_metadata=run_metadata)
        else:
            batches = self.figures.batches
            batch_results = self.sim.run(tasks, self.val_dict(subrange=batches), options=options, run_metadata=run_metadata)
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

    def view_batch(self, fieldname, subrange=None):
        iterator = self.view_train_iterator if self.value_view_training_data else self.val_iterator
        if isinstance(subrange, int):
            subrange = [subrange]
        result = iterator.get_batch([fieldname], subrange)[fieldname]
        iterator.progress()
        return result

    def save_model(self):
        dir = self.scene.subpath("checkpoint_%08d"%self.time)
        self.sim.save(dir)
        return dir

    def load_model(self, checkpoint_dir):
        self.sim.restore(checkpoint_dir, scope=self.model_scope_name)

    def model_scope(self):
        return tf.variable_scope(self.model_scope_name)

    def add_field(self, name, field):
        """

        :param name: field name
        :param field: Tensor, string (database fieldname) or function
        """
        if istensor(field):
            FieldSequenceModel.add_field(self, name, lambda: self.view(field))
        elif isinstance(field, six.string_types):
            FieldSequenceModel.add_field(self, name, lambda: self.view_batch(field))
        else:
            FieldSequenceModel.add_field(self, name, field)