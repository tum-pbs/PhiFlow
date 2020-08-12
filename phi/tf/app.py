import numpy as np
import phi.app.app as base_app
import six

from . import tf
from phi.app.app import EditableFloat, EditableInt, EditableValue
from phi.data.dataset import Dataset
from phi.data.reader import BatchReader
from phi.physics.field import Field, StaggeredGrid
from phi.tf.data import Dataset as TFDataset

from . import TF_BACKEND
from .session import Session
from .world import tf_bake_graph
from .. import math
from ..app import EditableBool


class App(base_app.App):

    def __init__(self, *args, **kwargs):
        base_app.App.__init__(self, *args, **kwargs)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = Session(self.scene, session=tf.Session(config=config))
        self.scalars = []
        self.scalar_names = []
        self.editable_placeholders = {}  # placeholder -> attribute name
        self.auto_bake = True
        self.add_trait('tensorflow')

    def prepare(self):
        if self.prepared:
            return
        base_app.App.prepare(self)
        if self.auto_bake:
            self.info("Baking static TensorFlow graph (disable by setting 'app.auto_bake = False'")
            tf_bake_graph(self.world, self.session)
        self.info('Initializing variables')
        self.session.initialize_variables()
        return self

    def add_scalar(self, name, node):
        assert TF_BACKEND.is_tensor(node), 'add_scalar requires a TensorFlow tensor but got %s' % node
        self.scalar_names.append(name)
        self.scalars.append(node)

    def editable_float(self, name, initial_value, minmax=None, log_scale=None):
        val = EditableFloat(name, initial_value, minmax, None, log_scale)
        self.set_editable_value(name, val)
        placeholder = tf.placeholder(TF_BACKEND.precision_dtype, (), name.lower().replace(' ', '_'))
        self.add_scalar(name, placeholder)
        self.editable_placeholders[placeholder] = name
        return placeholder

    def editable_int(self, name, initial_value, minmax=None):
        val = EditableInt(name, initial_value, minmax, None)
        self.set_editable_value(name, val)
        placeholder = tf.placeholder(tf.int32, (), name.lower().replace(' ', '_'))
        self.add_scalar(name, placeholder)
        self.editable_placeholders[placeholder] = name
        return placeholder

    def editable_bool(self, name, initial_value, add_as_scalar=False):
        val = EditableBool(name, initial_value)
        self.set_editable_value(name, val)
        placeholder = tf.placeholder(tf.bool, (), name.lower().replace(' ', '_'))
        if add_as_scalar:
            self.add_scalar(name, math.to_float(placeholder))
        self.editable_placeholders[placeholder] = name
        return placeholder

    def editable_values_dict(self):
        return {placeholder: self.get_editable_value(name) for placeholder, name in self.editable_placeholders.items()}

    def get_editable_value(self, name):
        value = getattr(self, '_ed_val_' + name.lower())
        if isinstance(value, EditableValue):
            return value.initial_value
        else:
            return value

    def set_editable_value(self, name, value):
        setattr(self, '_ed_val_' + name.lower(), value)


def EVERY_EPOCH(tfapp): return tfapp.steps % tfapp.epoch_size == 0


class LearningApp(App):

    def __init__(self, name='TensorFlow application', subtitle='',
                 learning_rate=1e-3,
                 training_batch_size=4,
                 validation_batch_size=16,
                 model_scope_name='model',
                 base_dir='~/phi/model/',
                 stride=None,
                 epoch_size=None,
                 force_custom_stride=False,
                 log_scalars=EVERY_EPOCH,
                 **kwargs):
        App.__init__(self, name=name, subtitle=subtitle, base_dir=base_dir, **kwargs)
        self.add_trait('model')

        # --- Model ---
        self.model_scope_name = model_scope_name
        self.auto_bake = False
        self.scalar_values = {}
        self.scalar_values_validation = {}
        self.learning_rate = self.editable_float('Learning_Rate', learning_rate)
        self.all_optimizers = []
        # --- Data ---
        self.training = tf.placeholder(tf.bool, (), 'training')
        self.training_batch_size = training_batch_size
        self.validation_batch_size = validation_batch_size
        self._placeholder_struct = None  # Data-placeholders or Iterator placeholder
        self._training_set = None
        self._validation_set = None
        self._pipeline = None
        self.set_data(None, None)
        assert stride is None or epoch_size is None
        self.epoch_size = epoch_size if epoch_size is not None else stride
        assert isinstance(log_scalars, bool) or callable(log_scalars)
        self.log_scalars = log_scalars

    def prepare(self):
        if self.prepared:
            return

        scalars = [tf.summary.scalar(self.scalar_names[i], self.scalars[i]) for i in range(len(self.scalars))]
        self.merged_scalars = tf.summary.merge(scalars)

        App.prepare(self)  # initializes global variables

        model_parameter_count = 0
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_scope_name):
            if not 'Adam' in var.name:
                model_parameter_count += int(np.prod(var.get_shape().as_list()))
                # if 'conv' in var.name and 'kernel' in var.name:
                #     tf.summary.image(var.name, var)
        self.add_custom_property('parameter_count', model_parameter_count)
        self.info('Model variables contain %d total parameters.' % model_parameter_count)
        # --- Use world.batch_size? ---
        if self.world.batch_size is not None:
            self.training_batch_size = self.world.batch_size
            self.validation_batch_size = self.world.batch_size
        # --- Epoch size ---
        if self.epoch_size is None:
            if self._train_reader is not None:
                self.epoch_size = len(self._train_reader.all_batches(batch_size=self.training_batch_size))
            else:
                self.epoch_size = 1
        self.sequence_stride = self.epoch_size
        # --- Validate ---
        self.validation_step()
        return self

    def set_learning_rate(self, learning_rate):
        self.set_editable_value('Learning_Rate', learning_rate)

    def set_data(self, dict, train=None, val=None):
        """
Specify what data to use for training and validation.

The content of `dict` determines the data pipeline that is used.
  - 'placeholder' pipline: the static TensorFlow graph uses placeholders as input. `dict` maps from placeholders to file names or Stream instances. Placeholders and corresponding streams may be placed inside structs.
  - 'dataset_handle' pipeline: Use TensorFlow data pipeline. `dict` contains 'iterator_handle' and related properties as returned by `build_graph_input(...)[1]`.

Regardless of pipeline, the recommended way to obtain `dict` is through `build_graph_input(...)[1]`.

        :param dict: pipeline-dependent dict
        :type dict: dict
        :param train: (optional) Dataset used for training
        :type train: Dataset
        :param val: (optional) Dataset used for validation
        :type val: Dataset
        """
        assert isinstance(train, Dataset) or train is None
        assert isinstance(val, Dataset) or val is None
        if train is not None or val is not None:
            assert dict is not None
        if train is not None and val is not None:
            self.value_view_training_data = False
        self._training_set = train
        self._validation_set = val
        if dict is not None and 'iterator_handle' in dict:
            self._init_tf_pipeline(**dict)
        else:
            self._init_numpy_iterators(dict)

    def _init_numpy_iterators(self, dict):
        self._pipeline = 'placeholder'
        self._placeholder_struct = []
        self._channel_struct = []
        if dict is not None:
            for key, value in dict.items():
                self._placeholder_struct.append(key)
                self._channel_struct.append(value)
        self._channel_struct = tuple(self._channel_struct)
        self._placeholder_struct = tuple(self._placeholder_struct)
        # Train
        if self._training_set is not None:
            self._train_reader = BatchReader(self._training_set, self._channel_struct)
            self._train_iterator = self._train_reader.all_batches(batch_size=self.world.batch_size or self.training_batch_size, loop=True)
        else:
            self._train_reader = None
            self._train_iterator = None
        # Val
        if self._validation_set is not None:
            self._val_reader = BatchReader(self._validation_set, self._channel_struct)
        else:
            self._val_reader = None

    def _init_tf_pipeline(self, iterator_handle, names, shapes, dtypes, frames):
        self._placeholder_struct = iterator_handle
        self._pipeline = 'dataset_handle'
        if self._training_set is not None:
            assert isinstance(self._training_set, TFDataset)
            if self._training_set.name is None:
                self._training_set.name = 'train'
            self._training_set.setup(names, shapes, dtypes, batch_size=self.training_batch_size, frames=frames)
            self._training_set.reset_iterator(self.session)
        if self._validation_set is not None:
            assert isinstance(self._validation_set, TFDataset)
            if self._validation_set.name is None:
                self._validation_set.name = 'validation'
            self._validation_set.setup(names, shapes, dtypes, batch_size=self.validation_batch_size, frames=frames)

    def add_objective(self, loss, name='Loss', optimizer=None, reg=None, vars=None):
        assert len(loss.shape) <= 1, 'Loss function must be a scalar'
        if not optimizer:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)

        if reg is not None:
            self.add_scalar(name + '_reg_unscaled', reg)
            reg_scale = self.editable_float(name + '_reg_scale', 1.0)
            optim_function = loss + reg * reg_scale
        else:
            optim_function = loss

        if isinstance(vars, six.string_types):
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vars)

        node = optimizer.minimize(optim_function, var_list=vars)

        self.add_scalar(name, loss)
        self.all_optimizers.append(node)
        return node

    def step(self):
        optimized = self.optimization_step(self.all_optimizers)
        if not optimized:
            self.steps -= 1
        if self._pipeline == 'placeholder' and self.steps % self.epoch_size == 0:
            self.validation_step(create_checkpoint=True)
        return self

    def optimization_step(self, optim_nodes, log_loss=None):
        try:
            optim_nodes = list(optim_nodes)
        except:
            optim_nodes = [optim_nodes]
        if self._pipeline == 'placeholder':
            batch = next(self._train_iterator) if self._train_iterator is not None else None
        elif self._pipeline == 'dataset_handle':
            batch = self._training_set.iterator_handle
        else:
            raise NotImplementedError('Pipeline %s' % self._pipeline)
        feed_dict = self._feed_dict(batch, True)
        try:
            scalar_values = self.session.run(optim_nodes + self.scalars, feed_dict, summary_key='train', merged_summary=self.merged_scalars, time=self.steps)[len(optim_nodes):]
        except tf.errors.OutOfRangeError as error:
            if self._pipeline != 'dataset_handle':
                raise error
            self.on_training_set_end()
            return False
        self.scalar_values = {name: value for name, value in zip(self.scalar_names, scalar_values)}
        if log_loss is None:
            log_loss = self.log_scalars
        if callable(log_loss):
            log_loss = log_loss(self)
        assert isinstance(log_loss, bool)
        if log_loss:
            self.info('Optimization (%06d): ' % self.steps + ', '.join([self.scalar_names[i] + ': ' + str(scalar_values[i]) for i in range(len(self.scalars))]))
        return True

    def on_training_set_end(self):
        self.validation_step(create_checkpoint=True)
        self._training_set.reset_iterator(self.session)

    def validation_step(self, create_checkpoint=False):
        if self._validation_set is None:
            return
        if self._pipeline == 'placeholder':
            batch = self._val_reader[0:self.validation_batch_size]
        elif self._pipeline == 'dataset_handle':
            batch = self._validation_set.get_reset_handle(self.session)
        else:
            raise NotImplementedError('Pipeline %s' % self._pipeline)
        feed_dict = self._feed_dict(batch, False)
        # ToDo iterate over complete valiadtion set and average the results, e.g. with tf.contrib.metrics.streaming_mean - https://stackoverflow.com/questions/40788785/how-to-average-summaries-over-multiple-batches
        scalar_values = self.session.run(self.scalars, feed_dict, summary_key='val', merged_summary=self.merged_scalars, time=self.steps)
        self.scalar_values_validation = {name: value for name, value in zip(self.scalar_names, scalar_values)}
        if create_checkpoint:
            self.save_model()
        self.info('Validation (%06d): ' % self.steps + ', '.join([self.scalar_names[i] + ': ' + str(scalar_values[i]) for i in range(len(self.scalars))]))

    def base_feed_dict(self):
        return {}

    def _feed_dict(self, batch, training):
        """
Assemble a complete feed dict for graph execution.
        :param batch: (optional)
          'placeholder' pipeline: struct of Numpy arrays matching self._placeholder_struct
          'dataset_handle' pipeline: iterator_handle string value
        :param training: bool
        :return: dict that can be passed to session.run()
        """
        feed_dict = self.base_feed_dict()
        feed_dict.update(self.editable_values_dict())
        feed_dict[self.training] = training
        if batch is not None:
            feed_dict[self._placeholder_struct] = batch
        return feed_dict

    @property
    def view_reader(self):
        if self._val_reader is None and self._train_reader is None:
            return None
        if self._val_reader is None:
            return self._train_reader
        return self._train_reader if self.value_view_training_data else self._val_reader

    def _view_dataset(self):
        if self._validation_set is None and self._training_set is None:
            return None
        if self._validation_set is None:
            return self._training_set
        if self.value_view_training_data:
            return self._training_set
        else:
            return self._validation_set

    def view(self, tasks):
        if tasks is None:
            return None
        if self._pipeline == 'placeholder':
            batch = self.view_reader[0:self.validation_batch_size] if self.view_reader is not None else None
        elif self._pipeline == 'dataset_handle':
            batch = self._view_dataset().get_reset_handle(self.session)
        else:
            raise NotImplementedError('Pipeline %s' % self._pipeline)
        feed_dict = self._feed_dict(batch, False)
        return self.session.run(tasks, feed_dict)

    @property
    def viewed_batch(self):
        assert self.view_reader is not None, 'There is no data to view.'
        return self.view_reader[0:self.validation_batch_size]

    def view_batch(self, get_attribute):
        batch = self.view_reader[0:self.validation_batch_size]
        return get_attribute(batch)

    def save_model(self):
        dir = self.scene.subpath('checkpoint_%08d' % self.steps)
        self.session.save(dir)
        return dir

    def action_save_model(self):
        self.save_model()

    def load_model(self, checkpoint_dir):
        self.session.restore(checkpoint_dir, scope=self.model_scope_name)

    def model_scope(self):
        return tf.variable_scope(self.model_scope_name)

    def add_field(self, name, field):
        """

        :param name: channel name
        :param field: Tensor, string (database fieldname) or function
        """
        if is_tensorflow_field(field):
            App.add_field(self, name, lambda: self.view(field))
        else:
            App.add_field(self, name, field)


TFApp = LearningApp


def is_tensorflow_field(obj):
    if TF_BACKEND.is_tensor(obj):
        return True
    if isinstance(obj, StaggeredGrid):
        return np.any([is_tensorflow_field(grid) for grid in obj.data])
    if isinstance(obj, Field):
        return TF_BACKEND.is_tensor(obj.data)
    return False
