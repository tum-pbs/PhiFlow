import contextlib
import logging
import os
import threading

import numpy as np
from . import tf
from phi import struct
from .profiling import Timeliner
from .util import isplaceholder, istensor


class Session(object):

    def __init__(self, scene, session=None):
        self._scene = scene
        self._session = session if session is not None else tf.Session()
        assert self._session.graph == tf.get_default_graph(), 'Session %s does not reference the current TensorFlow graph.'
        self.graph = tf.get_default_graph()
        self.summary_writers = {}
        self.summary_directory = os.path.abspath(scene.subpath('summary')) if scene is not None else None
        self.profiling_directory = scene.subpath("profile") if scene is not None else None
        self.trace_count = 0
        self.saver = None

    def initialize_variables(self):
        self._session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=100, allow_empty=True)

    def run(self, fetches, feed_dict=None, summary_key=None, time=None, merged_summary=None, item_condition=struct.ALL_ITEMS):
        if isinstance(fetches, np.ndarray):
            return fetches
        if fetches is None:
            return None

        tensor_feed_dict = None
        if feed_dict is not None:
            tensor_feed_dict = {}
            for (key, value) in feed_dict.items():
                pairs = struct.zip([key, value], item_condition=item_condition, zip_parents_if_incompatible=True)

                def add_to_dict(key_tensor, value_tensor):
                    if isplaceholder(key_tensor):
                        tensor_feed_dict[key_tensor] = value_tensor
                    return None
                struct.map(add_to_dict, pairs, item_condition=item_condition, content_type=struct.INVALID)

        tensor_fetches = struct.flatten(fetches, item_condition=item_condition)
        if isinstance(fetches, (tuple, list)):
            def is_fetch(x): return istensor(x) or _identity_in(x, fetches)
        else:
            def is_fetch(x): return istensor(x) or x is fetches
        tensor_fetches = tuple(filter(is_fetch, tensor_fetches))

        # Handle tracing
        trace = _trace_stack.get_default(raise_error=False)
        if trace:
            options = trace.timeliner.options
            run_metadata = trace.timeliner.run_metadata
        else:
            options = None
            run_metadata = None

        # Summary
        if summary_key is not None and merged_summary is not None:
            tensor_fetches = (merged_summary,) + tensor_fetches

        result_fetches = self._session.run(tensor_fetches, tensor_feed_dict, options, run_metadata)
        result_dict = {fetch: result for fetch, result in zip(tensor_fetches, result_fetches)}

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

        if trace:
            trace.timeliner.add_run()

        def replace_tensor_with_value(fetch):
            try:
                if fetch in result_dict:
                    return result_dict[fetch]
                else:
                    return fetch
            except TypeError:  # not hashable
                return fetch
        result = struct.map(replace_tensor_with_value, fetches, item_condition=item_condition)
        return result

    def profiler(self):
        os.path.isdir(self.profiling_directory) or os.makedirs(self.profiling_directory)
        self.trace_count += 1
        return Trace(self.trace_count, self.profiling_directory)

    def save(self, dir):
        assert self.saver is not None, "save() called before initialize_variables()"
        os.path.isdir(dir) or os.makedirs(dir)
        self.saver.save(self._session, os.path.join(dir, "model.ckpt"))

    def restore(self, dir, scope=None):
        path = os.path.join(dir, "model.ckpt")
        vars = self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        if len(vars) == 0:
            raise ValueError('The current graph does not contain any variables in scope "%s.\nAll: %s"' % (scope, self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
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
            logging.info(checkpoint_utils.list_variables(dir))
            raise e

    def as_default(self):
        return self._session.as_default()


class Trace(object):

    def __init__(self, index, directory):
        self.index = index
        self.directory = directory
        self.timeliner = None
        self.timeline_file = None
        self._default_simulation_context_manager = None

    def __enter__(self):
        self.timeline_file = os.path.join(self.directory, 'trace %d.json' % self.index)
        self.timeliner = Timeliner()

        if self._default_simulation_context_manager is None:
            self._default_simulation_context_manager = _trace_stack.get_controller(self)
        return self._default_simulation_context_manager.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timeliner.save(self.timeline_file)

        self._default_simulation_context_manager.__exit__(exc_type, exc_val, exc_tb)
        self._default_simulation_context_manager = None


class _TraceStack(threading.local):

    def __init__(self):
        self.stack = []

    def get_default(self, raise_error=True):
        if raise_error:
            assert len(self.stack) > 0, "Default simulation required. Use 'with simulation:' or 'with simulation.as_default():"
        return self.stack[-1] if len(self.stack) >= 1 else None

    def reset(self):
        self.stack = []

    def is_cleared(self):
        return not self.stack

    @contextlib.contextmanager
    def get_controller(self, default):
        """Returns a context manager for manipulating a default stack."""
        try:
            self.stack.append(default)
            yield default
        finally:
            # stack may be empty if reset() was called
            if self.stack:
                self.stack.remove(default)


_trace_stack = _TraceStack()


def _identity_in(obj, list):
    for item in list:
        if item is obj:
            return True
    return False
