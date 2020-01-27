from contextlib import contextmanager

import phi.app.app as base_app

from .torch_util import torch_to_numpy, torch_from_numpy


class App(base_app.App):

    def __init__(self, *args, **kwargs):
        base_app.App.__init__(self, *args, **kwargs)
        self.add_trait('torch')
        self.auto_convert = True

    def prepare(self):
        if self.prepared: return
        base_app.App.prepare(self)
        if self.auto_convert:
            self.world.state = torch_from_numpy(self.world.state)

    def add_scalar(self, name, node):
        pass

    def editable_float(self, name, initial_value, minmax=None, log_scale=None):
        return initial_value  # ToDo

    def editable_int(self, name, initial_value, minmax=None):
        return initial_value  # ToDo

    def editable_values_dict(self):
        raise NotImplementedError()

    def add_field(self, name, value):
        if callable(value):
            wrapped = lambda: torch_to_numpy(value())
        else:
            wrapped = torch_to_numpy(value)
        base_app.App.add_field(self, name, wrapped)



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
        self.learning_rate = self.editable_float('Learning_Rate', learning_rate)
        self.all_optimizers = []
        self.training_batch_size = training_batch_size
        self.validation_batch_size = validation_batch_size
        self.model_scope_name = model_scope_name
        self.scalar_values = {}
        self.scalar_values_validation = {}
        assert stride is None or epoch_size is None
        self.epoch_size = epoch_size if epoch_size is not None else stride
        assert isinstance(log_scalars, bool) or callable(log_scalars)
        self.log_scalars = log_scalars

    @contextmanager
    def model_scope(self):
        try:
            yield None
        finally:
            pass

    def add_objective(self, loss, name='Loss', optimizer=None, reg=None, vars=None):
        pass

    def step(self):
        pass