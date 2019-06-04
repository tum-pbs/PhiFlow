import numpy as np
from phi.math.container import *
from phi.math import load_tensorflow
from .profiling import *


class Session(object):

    def __init__(self, scene, session=tf.Session()):
        load_tensorflow()
        self._scene = scene
        self._session = session
        assert self._session.graph == tf.get_default_graph()
        self.graph = tf.get_default_graph()
        self.timeliner = None
        self.timeline_file = None
        self.summary_writers = {}
        self.summary_directory = scene.subpath('summary')
        self.profiling_directory = scene.subpath("profile")
        self.trace_count = 0
        self.saver = None

    def initialize_variables(self):
        import tensorflow as tf
        self._session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=100, allow_empty=True)

    def run(self, fetches, feed_dict=None, summary_key=None, time=None, merged_summary=None):
        if isinstance(fetches, np.ndarray):
            return fetches
        if fetches is None:
            return None
        single_task = not isinstance(fetches, (tuple,list))
        if single_task:
            fetches = [fetches]

        if feed_dict is not None:
            new_feed_dict = {}
            for (key, value) in feed_dict.items():
                key_tensors, _ = disassemble(key)
                value_tensors, _ = disassemble(value)
                for key_tensor, value_tensor in zip(key_tensors, value_tensors):
                    new_feed_dict[key_tensor] = value_tensor
            feed_dict = new_feed_dict

        tensor_fetches, reassemble = list_tensors(fetches)

        # Handle tracing
        if self.timeliner:
            options = self.timeliner.options
            run_metadata = self.timeliner.run_metadata
        else:
            options = None
            run_metadata = None

        # Summary
        if summary_key is not None and merged_summary is not None:
            tensor_fetches = [merged_summary] + tensor_fetches

        result_fetches = self._session.run(tensor_fetches, feed_dict, options, run_metadata)

        if summary_key:
            summary_buffer = result_fetches[0]
            result_fetches = result_fetches[1:]
            if summary_key in self.summary_writers:
                summary_writer = self.summary_writers[summary_key]
            else:
                summary_writer = tf.summary.FileWriter(os.path.join(self.summary_directory, str(summary_key)), self.graph)
                self.summary_writers[summary_key] = summary_writer
            summary_writer.add_summary(summary_buffer, time)
            summary_writer.flush()

        if self.timeliner:
            self.timeliner.add_run()

        result_fetches_containers = reassemble(result_fetches)
        if single_task:
            return result_fetches_containers[0]
        else:
            return result_fetches_containers

    @property
    def tracing(self):
        return self.timeliner is not None

    @tracing.setter
    def tracing(self, value):
        if (self.timeliner is not None) == value: return
        if value:
            os.path.isdir(self.profiling_directory) or os.makedirs(self.profiling_directory)
            self.trace_count += 1
            self.timeline_file = os.path.join(self.profiling_directory, 'trace %d.json' % self.trace_count)
            self.timeliner = Timeliner()
        else:
            self.timeliner.save(self.timeline_file)
            self.timeliner = None


    def save(self, dir):
        assert self.saver is not None, "save() called before initialize_variables()"
        os.path.isdir(dir) or os.makedirs(dir)
        self.saver.save(self._session, os.path.join(dir, "model.ckpt"))

    def restore(self, dir, scope=None):
        path = os.path.join(dir, "model.ckpt")
        vars = self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        saver = tf.train.Saver(var_list=vars)
        saver.restore(self._session, path)

    def restore_new_scope(self, dir, saved_scope, tf_scope):
        var_remap = dict()
        vars = [v for v in self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf_scope) if "Adam" not in v.name]
        for var in vars:
            var_remap[saved_scope + var.name[len(tf_scope):-2]] = var
        path = os.path.join(dir, "model.ckpt")
        saver = tf.train.Saver(var_list=var_remap)
        try:
            saver.restore(self._session, path)
        except tf.errors.NotFoundError as e:
            from tensorflow.contrib.framework.python.framework import checkpoint_utils
            print(checkpoint_utils.list_variables(dir))
            raise e