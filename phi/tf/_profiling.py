import json
import os
import threading

import tensorflow as tf
from tensorflow.python.client import timeline


class Timeliner:

    # _timeline_dict = None
    # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()

    def update_timeline(self, chrome_trace):
        # convert chrome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        os.path.isdir(os.path.dirname(f_name)) or os.makedirs(os.path.dirname(f_name))
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)

    def add_run(self, run_metadata=None):
        if run_metadata is None:
            run_metadata = self.run_metadata
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        self.update_timeline(chrome_trace)


def launch_tensorboard(log_dir, same_process=False, port=6006):
    if port is None:
        port = 6006
    if same_process:
        from tensorboard import main as tb
        tf.flags.FLAGS.logdir = log_dir
        tf.flags.FLAGS.reload_interval = 1
        tf.flags.FLAGS.port = port
        threading.Thread(target=tb.main).start()
    else:
        def run_tb():
            os.system('tensorboard --logdir=%s --port=%d' % (log_dir,port))
        threading.Thread(target=run_tb).start()
    try:
        import phi.local.hostname
        host = phi.local.hostname.hostname
    except (ImportError, AttributeError):
        host = 'localhost'  # socket.gethostname()
    url = "http://%s:%d/" % (host,port)
    return url
