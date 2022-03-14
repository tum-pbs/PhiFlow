import numpy
import jax
from jax import random
from jax.experimental import stax  # from jax.example_libraries import stax
from jax.experimental import optimizers as optim
from typing import Callable

from phi import math
from .. import JAX


class StaxNet:

    def __init__(self, initialize: Callable, apply: Callable, input_shape: tuple):
        self._initialize = initialize
        self._apply = apply
        self._input_shape = input_shape
        self.parameters = None

    def initialize(self):
        rnd_key = JAX.rnd_key
        JAX.rnd_key, init_key = random.split(rnd_key)
        out_shape, self.parameters = self._initialize(init_key, input_shape=self._input_shape)

    def __call__(self, *args, **kwargs):
        return self._apply(self.parameters, *args)


class JaxOptimizer:

    def __init__(self, initialize: Callable, update: Callable, get_params: Callable):
        self._initialize, self._update, self._get_params = initialize, update, get_params  # Stax functions
        self._state = None
        self._step_i = 0

    def initialize(self, net: tuple):
        self._state = self._initialize(net)

    def update(self, grads: tuple):
        self._state = self._update(self._step_i, list(grads[0]), self._state)
        self._step_i += 1

    def get_network_parameters(self):
        return self._get_params(self._state)



def parameter_count(model: StaxNet) -> int:
    """
    Counts the number of parameters in a model.

    Args:
        model: Stax model

    Returns:
        `int`
    """
    total = 0
    for layer in model.parameters:
        for parameter in layer:
            total += numpy.prod(parameter.shape)
    return int(total)


def save_state(obj: StaxNet or JaxOptimizer, path: str):
    """
    Write the state of a module or optimizer to a file.

    See Also:
        `load_state()`

    Args:
        obj: `torch.nn.Module or torch.optim.Optimizer`
        path: File path as `str`.
    """
    if not path.endswith('.npy'):
        path += '.npy'
    if isinstance(obj, StaxNet):
        numpy.save(path, obj.parameters)
    else:
        pass  # ToDo
        # numpy.save(path, obj._state)


def load_state(obj: StaxNet or JaxOptimizer, path: str):
    """
    Read the state of a module or optimizer from a file.

    See Also:
        `save_state()`

    Args:
        obj: `torch.nn.Module or torch.optim.Optimizer`
        path: File path as `str`.
    """
    if not path.endswith('.npy'):
        path += '.npy'
    if isinstance(obj, StaxNet):
        state = numpy.load(path, allow_pickle=True)
        obj.parameters = tuple([tuple(layer) for layer in state])
    else:
        pass  # ToDo


def update_weights(net: StaxNet, optimizer: JaxOptimizer, loss_function: Callable, *loss_args, **loss_kwargs):
    """
    Computes the gradients of `loss_function` w.r.t. the parameters of `net` and updates its weights using `optimizer`.

    This is the Jax version. Analogue functions exist for other learning frameworks.

    Args:
        net: Learning model.
        optimizer: Optimizer.
        loss_function: Loss function, called as `loss_function(*loss_args, **loss_kwargs)`.
        *loss_args: Arguments given to `loss_function`.
        **loss_kwargs: Keyword arguments given to `loss_function`.

    Returns:
        Output of `loss_function`.
    """

    def loss_depending_on_net(params: tuple):
        return loss_function(*loss_args, **loss_kwargs)

    value, grad = math.functional_gradient(loss_depending_on_net)(net.parameters)
    optimizer.update(grad)
    net.parameters = optimizer.get_network_parameters()
    return value


def adam(net: StaxNet, learning_rate: float = 1e-3, betas=(0.9, 0.999), epsilon=1e-07):
    """
    Creates an Adam optimizer for `net`, alias for [`jax.example_libraries.optimizers.adam`](https://jax.readthedocs.io/en/latest/jax.example_libraries.optimizers.html).
    Analogue functions exist for other learning frameworks.
    """
    opt = JaxOptimizer(*optim.adam(learning_rate, betas[0], betas[1], epsilon))
    opt.initialize(net.parameters)
    return opt


def dense_net(in_channels: int,
              out_channels: int,
              layers: tuple or list,
              batch_norm=False,
              activation='ReLU') -> StaxNet:
    if batch_norm:
        raise NotImplementedError("only batch_norm=False currently supported")
    layers = [in_channels, *layers, out_channels]
    activation = {'ReLU': stax.Relu, 'Sigmoid': stax.Sigmoid, 'tanh': stax.Tanh}[activation]
    stax_layers = []
    for neuron_count in layers:
        stax_layers.append(stax.Dense(neuron_count))
        stax_layers.append(activation)
        stax.Conv()
    stax_layers.append(stax.Dense(out_channels))
    net_init, net_apply = stax.serial(*stax_layers)
    net = StaxNet(net_init, net_apply, (-1, in_channels))
    net.initialize()
    return net


def u_net(in_channels: int,
          out_channels: int,
          levels: int = 4,
          filters: int or tuple or list = 16,
          batch_norm=True,
          activation='ReLU',
          in_spatial=None) -> StaxNet:
    # if not batch_norm:
    #     raise NotImplementedError("only batch_norm=True currently supported")
    # if isinstance(filters, (tuple, list)):
    #     assert len(filters) == levels, f"List of filters has length {len(filters)} but u-net has {levels} levels."
    # else:
    #     filters = (filters,) * levels
    # net = UNet(in_channels, out_channels, filters)
    # net = net.to(TORCH.get_default_device().ref)
    # return net
    activation = {'ReLU': stax.Relu, 'Sigmoid': stax.Sigmoid, 'tanh': stax.Tanh}[activation]
    net_init, net_apply = stax.serial(
        stax.GeneralConv(2, filters, (3, 3), strides=(1, 1), padding='SAME'), activation,
        stax.MaxPool((2, 2), padding='VALID'),
        stax.Dense(8), activation,
        stax.Dense(16), activation,
        stax.Dense(16), activation,
        stax.Dense(8), activation,
        stax.Dense(out_channels),
    )

    init_funs, apply_funs = zip(*layers)

    def net_init(rng, input_shape):
        params = []
        for init_fun in init_funs:
            rng, layer_rng = random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        return input_shape, params

    def net_apply(params, inputs, **kwargs):
        rng = kwargs.pop('rng', None)
        rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
        for fun, param, rng in zip(apply_funs, params, rngs):
            inputs = fun(param, inputs, rng=rng, **kwargs)
        return inputs

    net = StaxNet(net_init, net_apply, (-1, in_channels))
    net.initialize()
    return net
